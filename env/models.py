"""Pydantic data models for CostAwareToolEnv.

Three main types flow through the environment:
  OrchestratorAction      — agent → env  (what tool to call)
  OrchestratorObservation — env → agent  (what the agent sees)
  OrchestratorState       — env → server (full bookkeeping snapshot)

ToolResult is returned by every tool and attached to the observation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Canonical tool IDs — order matters for UI display; must match config.tool_costs keys.
TOOL_IDS = [
    "ceramic_search",   # web retrieval (most useful for HotpotQA)
    "wiki_lookup",      # Wikipedia summary (good for entity facts)
    "calculator",       # safe AST math eval (essential for MATH)
    "code_executor",    # sandboxed Python (HumanEval)
    "llm_reason",       # LLM chain-of-thought (GPQA)
    "commit",           # submit answer — always free
]


class OrchestratorAction(BaseModel):
    """One step of the agent's interaction with the environment.

    Fields used per tool:
      ceramic_search  → query
      wiki_lookup     → query
      calculator      → expression  (falls back to query if blank)
      code_executor   → code_snippet (falls back to query if blank)
      llm_reason      → query
      commit          → answer
    """
    tool_id: str
    query: str = ""
    expression: str = ""
    code_snippet: str = ""
    answer: str = ""
    metadata: Optional[Dict[str, Any]] = None


class ToolResult(BaseModel):
    """Output produced by one tool call.

    Attached to OrchestratorObservation.last_tool_result and also
    appended (as a string) to the context_window.
    """
    tool_id: str
    output: str = ""         # Human-readable result text
    cost: float = 0.0        # Budget units charged (set by environment)
    latency_s: float = 0.0   # Wall-clock seconds (set by environment)
    error: Optional[str] = None  # Non-None if the tool call failed


class OrchestratorObservation(BaseModel):
    """Everything the agent sees at the start of each step.

    Designed to be complete: the agent should be able to make an
    informed tool-selection decision using only this observation.
    """
    # ── Current question ────────────────────────────────────────────────────
    question: str                               # Full question text
    question_idx: int                           # Position in the episode (0-indexed)
    domain: str                                 # "hotpotqa" | "math" | "gpqa" | "humaneval"
    question_embedding: List[float] = Field(default_factory=list)  # Optional embedding vector

    # ── Budget ──────────────────────────────────────────────────────────────
    total_budget: float
    budget_spent: float
    budget_remaining: float
    budget_remaining_ratio: float               # budget_remaining / total_budget ∈ [0, 1]

    # ── Progress on the current question ────────────────────────────────────
    tools_used_this_question: List[str] = Field(default_factory=list)
    steps_used_this_question: int = 0
    max_steps_per_question: int = 8
    last_tool_result: Optional[ToolResult] = None
    context_window: List[str] = Field(default_factory=list)  # "[tool_id] output" strings

    # ── Episode-level progress ───────────────────────────────────────────────
    step_idx: int = 0                           # Global step counter
    questions_remaining: int = 0                # Questions not yet started
    questions_answered: int = 0                 # Questions that received a commit
    accuracy_so_far: float = 0.0                # Running correctness rate

    # ── Terminal signals ─────────────────────────────────────────────────────
    question_done: bool = False                 # This question just ended (commit or max_steps)
    done: bool = False                          # Episode is over
    reward: Optional[float] = None              # Reward from the *previous* step


class OrchestratorState(BaseModel):
    """Full bookkeeping snapshot — returned alongside observation for logging.

    Contains all fields needed to reconstruct the episode history without
    digging into the environment's internal attributes.
    """
    episode_id: str
    total_budget: float
    budget_spent: float = 0.0
    questions_answered: int = 0
    total_correct: int = 0
    current_accuracy: float = 0.0
    budget_remaining_ratio: float = 1.0
    current_question_idx: int = 0
    current_question_steps: int = 0
    step_count: int = 0
