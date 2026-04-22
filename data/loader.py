"""Multi-domain dataset loader for ToolOrchestratorEnv.

Returns a flat list of question dicts, each with a 'domain' key.
Adapted from ToolOrchestratorEnv/scripts/process_datasets.py.
"""
from __future__ import annotations

import random
import re
import string
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# HuggingFace loader helper
# ---------------------------------------------------------------------------

def _hf_load(repo_id: str, config: Optional[str], split: str):
    import datasets as hf
    kwargs: Dict[str, Any] = {"split": split, "trust_remote_code": True}
    if config:
        kwargs["name"] = config
    return hf.load_dataset(repo_id, **kwargs)


# ---------------------------------------------------------------------------
# MATH (levels 3-5)
# ---------------------------------------------------------------------------

def _extract_boxed(solution: str):
    for cmd in ("boxed", "fbox"):
        marker = f"\\{cmd}" + "{"
        start = solution.rfind(marker)
        if start == -1:
            continue
        idx = start + len(marker) - 1
        depth = 0
        for i in range(idx, len(solution)):
            if solution[i] == "{":
                depth += 1
            elif solution[i] == "}":
                depth -= 1
                if depth == 0:
                    return solution[i + 1 - (i - idx):i].strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in solution.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _load_math(split: str, max_rows: int) -> List[Dict]:
    candidates = [
        ("DigitalLearningGmbH/MATH-lighteval", "default", "train"),
        ("lighteval/MATH-Hard", "default", "train"),
        ("hendrycks/competition_math", None, "train"),
    ]
    dataset = None
    for repo_id, cfg, spl in candidates:
        try:
            dataset = _hf_load(repo_id, cfg, spl)
            break
        except Exception:
            continue
    if dataset is None:
        return []

    rows = []
    for ex in dataset:
        level_text = str(ex.get("level", ""))
        m = re.search(r"(\d+)", level_text)
        if not m or int(m.group(1)) not in (3, 4, 5):
            continue
        answer = _extract_boxed(str(ex.get("solution", "")))
        rows.append({
            "question": str(ex.get("problem", "")).strip(),
            "answer": answer,
            "domain": "math",
            "difficulty": m.group(1),
            "subject": str(ex.get("type", "")),
            "source": "math",
        })
        if len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def _load_hotpotqa(split: str, max_rows: int) -> List[Dict]:
    hf_split = "train" if split in ("train", "validation") else split
    dataset = None
    for cfg in ("distractor", "fullwiki"):
        try:
            dataset = _hf_load("hotpotqa/hotpot_qa", cfg, hf_split)
            break
        except Exception:
            continue
    if dataset is None:
        return []

    subset = dataset.shuffle(seed=42).select(range(min(max_rows, len(dataset))))
    rows = []
    for ex in subset:
        rows.append({
            "question": str(ex.get("question", "")).strip(),
            "answer": str(ex.get("answer", "")).strip(),
            "domain": "hotpotqa",
            "difficulty": str(ex.get("level", "")),
            "type": str(ex.get("type", "")),
            "source": "hotpotqa",
        })
    return rows


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

def _resolve_gpqa_answer(ex: Dict) -> str:
    val = str(ex.get("Correct Answer", "")).strip()
    if val.upper() in {"A", "B", "C", "D"}:
        mapping = {
            "A": str(ex.get("Answer A", "")),
            "B": str(ex.get("Answer B", "")),
            "C": str(ex.get("Answer C", "")),
            "D": str(ex.get("Answer D", "")),
        }
        return mapping.get(val.upper(), val).strip()
    return val


def _load_gpqa(split: str, max_rows: int) -> List[Dict]:
    dataset = None
    for repo in ("Idavidrein/gpqa", "Wanfq/gpqa"):
        for cfg in ("gpqa_diamond", "gpqa_main"):
            try:
                dataset = _hf_load(repo, cfg, "train")
                break
            except Exception:
                continue
        if dataset is not None:
            break
    if dataset is None:
        return []

    rows = []
    for ex in dataset:
        answer = _resolve_gpqa_answer(ex)
        rows.append({
            "question": str(ex.get("Question", "")).strip(),
            "answer": answer,
            "domain": "gpqa",
            "difficulty": "graduate",
            "source": "gpqa",
        })
        if len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# HumanEval
# ---------------------------------------------------------------------------

def _load_humaneval(split: str, max_rows: int) -> List[Dict]:
    dataset = None
    for repo in ("openai/openai_humaneval", "openai/human-eval"):
        try:
            dataset = _hf_load(repo, None, "test")
            break
        except Exception:
            continue
    if dataset is None:
        return []

    rows = []
    for ex in dataset:
        rows.append({
            "question": str(ex.get("prompt", "")).strip(),
            "answer": str(ex.get("canonical_solution", "")).strip(),
            "domain": "humaneval",
            "difficulty": "code",
            "task_id": str(ex.get("task_id", "")),
            "test": str(ex.get("test", "")),
            "entry_point": str(ex.get("entry_point", "")),
            "source": "humaneval",
        })
        if len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Synthetic fallback (offline / CI)
# ---------------------------------------------------------------------------

_SYNTHETIC_TEMPLATES = [
    ("What is {a} + {b}?", "{c}", "math"),
    ("Who wrote {work}?", "{author}", "hotpotqa"),
    ("Solve for x: {a}x + {b} = {c}", "{x}", "math"),
    ("What is the capital of {country}?", "{capital}", "hotpotqa"),
]

_SYNTHETIC_DATA = [
    {"a": 12, "b": 7, "c": 19, "work": "Hamlet", "author": "Shakespeare",
     "country": "France", "capital": "Paris", "x": 3},
    {"a": 25, "b": 13, "c": 38, "work": "1984", "author": "George Orwell",
     "country": "Germany", "capital": "Berlin", "x": 5},
    {"a": 100, "b": 44, "c": 144, "work": "The Odyssey", "author": "Homer",
     "country": "Japan", "capital": "Tokyo", "x": 7},
]


def _synthetic_questions(n: int) -> List[Dict]:
    rows = []
    for i in range(n):
        tmpl, ans_tmpl, domain = _SYNTHETIC_TEMPLATES[i % len(_SYNTHETIC_TEMPLATES)]
        data = _SYNTHETIC_DATA[i % len(_SYNTHETIC_DATA)]
        try:
            question = tmpl.format(**data)
            answer = ans_tmpl.format(**data)
        except KeyError:
            question = f"Synthetic question {i}"
            answer = f"answer_{i}"
        rows.append({
            "question": question,
            "answer": str(answer),
            "domain": domain,
            "difficulty": "easy",
            "source": "synthetic",
        })
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_LOADERS = {
    "hotpotqa": _load_hotpotqa,
    "math":     _load_math,
    "gpqa":     _load_gpqa,
    "humaneval": _load_humaneval,
}


def load_all(split: str = "validation", max_per_domain: int = 200) -> List[Dict]:
    """Load all four domains and return a flat list with 'domain' keys.

    Falls back to synthetic questions if a domain is unavailable.
    """
    all_questions: List[Dict] = []
    for domain, loader_fn in _LOADERS.items():
        try:
            rows = loader_fn(split, max_per_domain)
            if rows:
                all_questions.extend(rows)
                print(f"[loader] {domain}: {len(rows)} questions")
            else:
                raise ValueError("empty")
        except Exception as exc:
            print(f"[loader] {domain} unavailable ({exc}), using synthetic fallback")
            synth = _synthetic_questions(max(5, max_per_domain // 10))
            for q in synth:
                q["domain"] = domain
            all_questions.extend(synth)

    random.shuffle(all_questions)
    return all_questions
