"""Cheapest-first baseline — always calls the cheapest available tool first."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader import load_all
from env.config import EnvConfig
from env.environment import CostAwareToolEnvironment
from env.models import OrchestratorAction
from tools import build_tool_registry


class CheapestFirstBaseline:
    """Calls tools in ascending cost order, commits after exhausting budget."""

    def __init__(self, config: EnvConfig):
        # Sort non-commit tools by cost
        self._order = sorted(
            [t for t in config.tool_costs if t != "commit"],
            key=lambda t: config.tool_costs[t],
        )
        self._commit_after = config.max_steps_per_question - 1
        self._steps_on_q = 0

    def get_action(self, obs) -> OrchestratorAction:
        if self._steps_on_q >= self._commit_after:
            self._steps_on_q = 0
            return OrchestratorAction(tool_id="commit", answer="I don't know")
        tool = self._order[self._steps_on_q % len(self._order)]
        self._steps_on_q += 1
        query = obs.question[:100] if hasattr(obs, "question") else ""
        return OrchestratorAction(tool_id=tool, query=query)

    def reset(self):
        self._steps_on_q = 0


def run_episode(seed: int = 0) -> dict:
    config = EnvConfig(num_questions=5, total_budget=30.0, seed=seed)
    tools = build_tool_registry(config)
    dataset = load_all(max_per_domain=20)
    env = CostAwareToolEnvironment(config=config, tools=tools, dataset=dataset)
    agent = CheapestFirstBaseline(config)

    obs, state = env.reset(seed=seed)
    agent.reset()
    total_reward = 0.0
    done = False
    while not done:
        action = agent.get_action(obs)
        if action.tool_id == "commit":
            agent.reset()
        obs, reward, done, state = env.step(action)
        total_reward += reward

    return {
        "total_reward": total_reward,
        "accuracy": state.current_accuracy,
        "budget_used": state.budget_spent,
        "questions_answered": state.questions_answered,
    }


if __name__ == "__main__":
    result = run_episode(seed=42)
    print("CheapestFirstBaseline:", result)
