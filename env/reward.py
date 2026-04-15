"""Reward functions for CostAwareToolEnv.

Adapted from SearchEconomicsEnv/env/reward.py, generalised to a multi-tool
action space where each tool has its own cost.

Two reward signals:

  step_reward   — small negative penalty charged every time the agent calls
                  a tool (including commit = 0 cost).  Discourages wasted
                  calls without forbidding exploration.

  commit_reward — composite reward awarded when the agent submits an answer.
                  Balances answer quality against remaining budget (Weitzman
                  style: you earn a bonus for being both correct *and* frugal).
"""
from __future__ import annotations

from .config import EnvConfig


def step_reward(tool_id: str, config: EnvConfig) -> float:
    """Return the (negative) cost of calling tool_id.

    Example: calculator → -0.1,  llm_reason → -2.0,  commit → 0.0
    """
    return -config.tool_costs.get(tool_id, 0.0)


def commit_reward(
    quality: float,
    budget_remaining_ratio: float,
    config: EnvConfig,
) -> float:
    """Composite reward on commit.

    Formula
    -------
        base  = incorrect_reward + quality × (correct_reward − incorrect_reward)
        η     = 1  if quality ≥ efficiency_bonus_threshold, else 0
        bonus = η × efficiency_bonus_weight × budget_remaining_ratio
        R     = base + bonus

    The efficiency bonus (bonus) is only non-zero when the agent both answers
    correctly (quality above threshold) *and* conserves budget.  This creates
    a soft incentive to use cheaper tools and commit early when confident.

    Parameters
    ----------
    quality               : float in [0, 1] — max(ExactMatch, TokenF1)
    budget_remaining_ratio: float in [0, 1] — fraction of budget still unspent
    config                : EnvConfig
    """
    q     = max(0.0, min(1.0, quality))
    base  = config.incorrect_reward + q * (config.correct_reward - config.incorrect_reward)
    eta   = 1.0 if q >= config.efficiency_bonus_threshold else 0.0
    bonus = eta * config.efficiency_bonus_weight * budget_remaining_ratio
    return base + bonus
