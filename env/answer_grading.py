"""Answer grading utilities: exact match + token F1.

Ported from SearchEconomicsEnv/env/answer_grading.py and adapted for
multi-domain use (HotpotQA-style EM/F1 + code/math fallback).
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Tuple


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> list[str]:
    """Lowercase, strip articles/punctuation, tokenise."""
    text = text.lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred)
    gold_tokens = normalize_answer(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    """Pull the answer string out of various agent output formats."""
    # Strip markdown fences
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()

    # Try JSON {"answer": ...}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("answer", "Answer", "result", "Result"):
                if key in parsed:
                    return str(parsed[key]).strip()
    except (json.JSONDecodeError, ValueError):
        pass

    # Prefix patterns
    for prefix in ("Answer:", "Final answer:", "Result:", "Output:"):
        idx = raw.lower().find(prefix.lower())
        if idx != -1:
            return raw[idx + len(prefix):].strip().split("\n")[0].strip()

    # Last non-empty line
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return lines[-1] if lines else raw.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def grade(predicted: str, ground_truth: str) -> Tuple[bool, float, float]:
    """Return (exact_match, f1, quality) where quality ∈ [0, 1]."""
    extracted = extract_answer(predicted)
    em = exact_match(extracted, ground_truth)
    f1 = token_f1(extracted, ground_truth)
    quality = 1.0 if em else f1
    return em, f1, quality
