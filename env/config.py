"""Configuration for CostAwareToolEnv.

All tuneable parameters live here so training scripts, the server, and
baselines all read from a single source of truth.  Override individual
fields in /reset via config_overrides, or subclass for experiment sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EnvConfig:
    # ── Episode structure ────────────────────────────────────────────────────
    total_budget: float = 50.0          # Total cost units for the whole episode
    num_questions: int = 10             # Questions drawn per episode
    max_steps_per_question: int = 8     # Auto-commit after this many tool calls
    data_split: str = "validation"      # HuggingFace dataset split to load
    seed: Optional[int] = None          # Global RNG seed (None = random)
    shuffle_questions: bool = True      # Shuffle sampled questions each episode

    # ── Domain mix ──────────────────────────────────────────────────────────
    # Fraction of questions drawn from each dataset. Must sum to ~1.0.
    domain_mix: Dict[str, float] = field(default_factory=lambda: {
        "hotpotqa":  0.4,   # Multi-hop factual QA
        "math":      0.3,   # Competition math (levels 3-5)
        "gpqa":      0.2,   # Graduate-level science
        "humaneval": 0.1,   # Python programming tasks
    })

    # ── Tool costs ──────────────────────────────────────────────────────────
    # Budget units consumed per tool call.  Commit is always free.
    tool_costs: Dict[str, float] = field(default_factory=lambda: {
        "ceramic_search": 1.0,
        "wiki_lookup":    0.5,
        "calculator":     0.1,
        "code_executor":  0.3,
        "llm_reason":     2.0,
        "commit":         0.0,
    })

    # ── Reward shaping ───────────────────────────────────────────────────────
    correct_reward: float = 1.0             # Base reward for a correct commit
    incorrect_reward: float = -0.5          # Base reward for a wrong commit
    efficiency_bonus_weight: float = 0.1    # γ: scales the efficiency bonus
    efficiency_bonus_threshold: float = 0.5 # Minimum quality to earn the bonus

    # ── Grading ─────────────────────────────────────────────────────────────
    # "em_only"  → only exact match counts as correct
    # "em_or_f1" → token F1 ≥ f1_count_threshold also counts as correct
    grade_count_correct_mode: str = "em_or_f1"
    f1_count_threshold: float = 0.5
