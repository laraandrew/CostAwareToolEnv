"""Core RL environment: ToolOrchestratorEnvironment.

Step logic:
  - Agent receives an OrchestratorObservation with the current question,
    budget, context, and available tools.
  - Agent picks a tool_id and optional query / code_snippet / answer.
  - Environment dispatches to the appropriate tool, charges cost, appends
    result to context_window, and returns the next observation + reward.
  - Episode ends when budget is exhausted OR all questions are answered.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .answer_grading import grade
from .config import EnvConfig
from .models import (
    OrchestratorAction,
    OrchestratorObservation,
    OrchestratorState,
    ToolResult,
    TOOL_IDS,
)
from .reward import commit_reward, step_reward


class ToolOrchestratorEnvironment:
    """
    OpenEnv-compatible RL environment for multi-tool cost-aware QA.

    Supports external tool injection so the server can wire in live
    Ceramic, code executor, etc.  Tools are callables with signature:
        tool_fn(action: OrchestratorAction) -> ToolResult
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        tools: Optional[Dict[str, Any]] = None,
        dataset: Optional[List[Dict[str, Any]]] = None,
    ):
        self.config  = config or EnvConfig()
        self.tools   = tools or {}      # tool_id -> callable
        self.dataset = dataset or []    # List of {question, answer, domain, ...}
        self._state: Optional[OrchestratorState] = None
        self._questions: List[Dict[str, Any]] = []
        self._current_q_idx: int = 0
        self._context_window: List[str] = []
        self._tools_used_this_q: List[str] = []
        self._steps_this_q: int = 0
        self._episode_done: bool = False

    # -----------------------------------------------------------------------
    # Reset
    # -----------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[OrchestratorObservation, OrchestratorState]:
        import random
        effective_seed = seed if seed is not None else self.config.seed
        rng = random.Random(effective_seed)

        questions = _sample_questions(self.dataset, self.config, rng)
        self._questions = questions
        self._current_q_idx = 0
        self._episode_done = False

        self._state = OrchestratorState(
            episode_id=str(uuid.uuid4()),
            total_budget=self.config.total_budget,
            budget_spent=0,
            questions_answered=0,
            total_correct=0,
            current_accuracy=0.0,
            budget_remaining_ratio=1.0,
            current_question_idx=0,
            current_question_steps=0,
            step_count=0,
        )

        self._context_window = []
        self._tools_used_this_q = []
        self._steps_this_q = 0

        obs = self._make_obs(reward=None, question_done=False, done=False)
        return obs, self._state

    # -----------------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------------

    def step(
        self, action: OrchestratorAction
    ) -> Tuple[OrchestratorObservation, float, bool, OrchestratorState]:
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() first.")

        tool_id = action.tool_id
        if tool_id not in TOOL_IDS:
            raise ValueError(f"Unknown tool_id: {tool_id!r}. Valid: {TOOL_IDS}")

        state   = self._state
        config  = self.config

        if self._current_q_idx >= len(self._questions):
            self._episode_done = True
            raise RuntimeError("Episode is done. Call reset() first.")

        q_entry = self._questions[self._current_q_idx]
        gold    = q_entry["answer"]

        # ---- Commit ---------------------------------------------------
        if tool_id == "commit":
            raw_pred = action.answer or ""
            em, f1, quality = grade(raw_pred, gold)

            r = commit_reward(
                quality=quality,
                budget_remaining_ratio=state.budget_remaining_ratio,
                config=config,
            )

            is_correct = (
                em if config.grade_count_correct_mode == "em_only"
                else (em or f1 >= config.f1_count_threshold)
            )
            state.questions_answered += 1
            state.total_correct += int(is_correct)
            state.current_accuracy = state.total_correct / state.questions_answered

            self._current_q_idx += 1
            self._context_window = []
            self._tools_used_this_q = []
            self._steps_this_q = 0

            episode_done = (
                self._current_q_idx >= len(self._questions)
                or state.budget_spent >= state.total_budget
            )
            self._episode_done = episode_done
            state.current_question_idx = self._current_q_idx
            state.current_question_steps = 0

            obs = self._make_obs(
                reward=r,
                question_done=True,
                done=episode_done,
                last_tool_result=ToolResult(
                    tool_id="commit", cost=0,
                    output=f"EM={em} F1={f1:.3f} quality={quality:.3f}"
                ),
            )
            return obs, r, episode_done, state

        # ---- Tool call ------------------------------------------------
        cost = config.tool_costs.get(tool_id, 0)
        budget_after = state.budget_spent + cost

        if budget_after > state.total_budget:
            r = config.incorrect_reward
            self._episode_done = True
            obs = self._make_obs(reward=r, question_done=True, done=True)
            return obs, r, True, state

        t0 = time.perf_counter()
        tool_fn = self.tools.get(tool_id)
        if tool_fn is None:
            tool_result = ToolResult(
                tool_id=tool_id, cost=cost,
                output="[Tool not available in this environment]",
                latency_s=0.0,
                error="not_available",
            )
        else:
            try:
                tool_result = tool_fn(action)
                tool_result.cost = cost
            except Exception as exc:
                tool_result = ToolResult(
                    tool_id=tool_id, cost=cost,
                    output=f"[Error: {exc}]",
                    latency_s=time.perf_counter() - t0,
                    error=str(exc),
                )
        tool_result.latency_s = time.perf_counter() - t0

        state.budget_spent = budget_after
        state.budget_remaining_ratio = max(
            0.0, (state.total_budget - state.budget_spent) / state.total_budget
        )
        state.step_count += 1
        state.current_question_steps += 1
        self._steps_this_q += 1
        self._tools_used_this_q.append(tool_id)
        self._context_window.append(f"[{tool_id}] {tool_result.output}")

        r = step_reward(tool_id, config)

        question_done = self._steps_this_q >= config.max_steps_per_question
        episode_done  = (
            state.budget_spent >= state.total_budget
            or (question_done and self._current_q_idx + 1 >= len(self._questions))
        )
        if question_done and not episode_done:
            self._current_q_idx += 1
            state.questions_answered += 1
            self._context_window = []
            self._tools_used_this_q = []
            self._steps_this_q = 0
            state.current_question_idx = self._current_q_idx
            state.current_question_steps = 0

        self._episode_done = episode_done

        obs = self._make_obs(
            reward=r,
            question_done=question_done,
            done=episode_done,
            last_tool_result=tool_result,
        )
        return obs, r, episode_done, state

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _make_obs(
        self,
        reward: Optional[float],
        question_done: bool,
        done: bool,
        last_tool_result: Optional[ToolResult] = None,
    ) -> OrchestratorObservation:
        state = self._state
        cfg   = self.config

        if 0 <= self._current_q_idx < len(self._questions):
            q_entry = self._questions[self._current_q_idx]
        elif self._questions:
            q_entry = self._questions[-1]
        else:
            q_entry = {"question": "", "answer": "", "domain": ""}

        return OrchestratorObservation(
            question=q_entry.get("question", ""),
            question_idx=self._current_q_idx,
            domain=q_entry.get("domain", ""),
            question_embedding=[],
            total_budget=cfg.total_budget,
            budget_spent=state.budget_spent,
            budget_remaining=state.total_budget - state.budget_spent,
            budget_remaining_ratio=state.budget_remaining_ratio,
            tools_used_this_question=list(self._tools_used_this_q),
            steps_used_this_question=self._steps_this_q,
            max_steps_per_question=cfg.max_steps_per_question,
            last_tool_result=last_tool_result,
            context_window=list(self._context_window),
            step_idx=state.step_count,
            questions_remaining=max(0, len(self._questions) - self._current_q_idx - 1),
            questions_answered=state.questions_answered,
            accuracy_so_far=state.current_accuracy,
            question_done=question_done,
            done=done,
            reward=reward,
        )


# ---------------------------------------------------------------------------
# Dataset sampling helper
# ---------------------------------------------------------------------------

def _sample_questions(
    dataset: List[Dict[str, Any]],
    config: EnvConfig,
    rng: Any,
) -> List[Dict[str, Any]]:
    """Sample `config.num_questions` questions according to domain_mix."""
    by_domain: Dict[str, List[Dict]] = {}
    for item in dataset:
        d = item.get("domain", "hotpotqa")
        by_domain.setdefault(d, []).append(item)

    selected = []
    for domain, frac in config.domain_mix.items():
        n = round(config.num_questions * frac)
        pool = by_domain.get(domain, [])
        if pool and n > 0:
            selected.extend(rng.sample(pool, min(n, len(pool))))

    if len(selected) < config.num_questions and dataset:
        remaining = [d for d in dataset if d not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: config.num_questions - len(selected)])

    if config.shuffle_questions:
        rng.shuffle(selected)

    return selected[: config.num_questions]
