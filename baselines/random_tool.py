"""Random tool baseline — picks uniformly from available tools each step."""
from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.loader import load_all
from env.config import EnvConfig
from env.environment import ToolOrchestratorEnvironment
from env.models import TOOL_IDS, OrchestratorAction
from tools import build_tool_registry

_NON_COMMIT = [t for t in TOOL_IDS if t != "commit"]


class RandomToolBaseline:
    """Selects a random tool each step; commits after max_steps_per_question - 1 steps."""

    def __init__(self, commit_after: int = 3):
        self.commit_after = commit_after
        self._steps_on_q = 0

    def get_action(self, obs) -> OrchestratorAction:
        if self._steps_on_q >= self.commit_after:
            self._steps_on_q = 0
            return OrchestratorAction(tool_id="commit", answer="I don't know")
        self._steps_on_q += 1
        tool = random.choice(_NON_COMMIT)
        query = obs.question[:100] if hasattr(obs, "question") else ""
        return OrchestratorAction(tool_id=tool, query=query)

    def reset(self):
        self._steps_on_q = 0


def run_episode(seed: int = 0) -> dict:
    config = EnvConfig(num_questions=5, total_budget=30.0, seed=seed)
    tools = build_tool_registry(config)
    dataset = load_all(max_per_domain=20)
    env = ToolOrchestratorEnvironment(config=config, tools=tools, dataset=dataset)
    agent = RandomToolBaseline(commit_after=config.max_steps_per_question - 1)

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
    print("RandomToolBaseline:", result)
