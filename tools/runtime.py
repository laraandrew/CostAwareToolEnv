"""Shared tool catalog and dispatch helpers for ToolOrchestratorEnv.

This module keeps the tool contract explicit:
  - the catalog describes every tool, its purpose, and its input field
  - registry validation catches config drift early
  - dispatch normalizes failures into ToolResult objects instead of
    letting exceptions leak through the environment loop
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping

from env.config import EnvConfig
from env.models import OrchestratorAction, TOOL_IDS, ToolResult


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    label: str
    purpose: str
    input_field: str
    cost: float
    notes: str


_TOOL_SPEC_TEMPLATES: Dict[str, Dict[str, str]] = {
    "ceramic_search": {
        "label": "Ceramic web search",
        "purpose": "Web retrieval for multi-hop factual QA",
        "input_field": "query",
        "notes": "Falls back to deterministic offline search when Ceramic credentials are unavailable.",
    },
    "wiki_lookup": {
        "label": "Wikipedia lookup",
        "purpose": "Entity facts, definitions, and short summaries",
        "input_field": "query",
        "notes": "Returns an explicit not-found or HTTP error result instead of crashing.",
    },
    "calculator": {
        "label": "Calculator",
        "purpose": "Arithmetic and symbolic math",
        "input_field": "expression",
        "notes": "Uses a restricted AST evaluator with comparisons and common math functions.",
    },
    "code_executor": {
        "label": "Python executor",
        "purpose": "HumanEval-style coding tasks",
        "input_field": "code_snippet",
        "notes": "Sandboxed exec with blocked imports, dunder attribute access, and unsafe builtins.",
    },
    "llm_reason": {
        "label": "LLM reasoning",
        "purpose": "Costly model-backed reasoning on hard problems",
        "input_field": "query",
        "notes": "Returns a clear no_api_key error when Together is unavailable.",
    },
    "commit": {
        "label": "Commit answer",
        "purpose": "Submit the final answer and advance the episode",
        "input_field": "answer",
        "notes": "Pass-through only; grading happens inside the environment.",
    },
}


def validate_tool_costs(config: EnvConfig) -> None:
    """Fail fast if the configured cost map drifts from the canonical tool set."""
    missing = [tool_id for tool_id in TOOL_IDS if tool_id not in config.tool_costs]
    if missing:
        raise ValueError(f"EnvConfig.tool_costs is missing required tools: {missing}")

    negative = {tool_id: cost for tool_id, cost in config.tool_costs.items() if cost < 0}
    if negative:
        raise ValueError(f"Tool costs must be non-negative: {negative}")


def build_tool_catalog(config: EnvConfig | None = None) -> List[ToolSpec]:
    """Return the canonical ordered catalog used by the UI and docs."""
    cfg = config or EnvConfig()
    validate_tool_costs(cfg)

    catalog: List[ToolSpec] = []
    for tool_id in TOOL_IDS:
        template = _TOOL_SPEC_TEMPLATES[tool_id]
        catalog.append(
            ToolSpec(
                tool_id=tool_id,
                label=template["label"],
                purpose=template["purpose"],
                input_field=template["input_field"],
                cost=cfg.tool_costs[tool_id],
                notes=template["notes"],
            )
        )
    return catalog


def build_tool_registry(config: EnvConfig | None = None) -> Dict[str, Callable[[OrchestratorAction], ToolResult]]:
    """Return the canonical mapping of tool_id -> tool callable."""
    from .calculator import calculator_tool
    from .ceramic_search import make_search_tool
    from .code_executor import code_executor_tool
    from .commit import commit_tool
    from .llm_reason import llm_reason_tool
    from .wiki_lookup import wiki_lookup_tool

    cfg = config or EnvConfig()
    validate_tool_costs(cfg)

    return {
        "ceramic_search": make_search_tool(),
        "calculator": calculator_tool,
        "wiki_lookup": wiki_lookup_tool,
        "code_executor": code_executor_tool,
        "llm_reason": llm_reason_tool,
        "commit": commit_tool,
    }


def dispatch_tool(
    tool_id: str,
    action: OrchestratorAction,
    registry: Mapping[str, Callable[[OrchestratorAction], ToolResult]],
) -> ToolResult:
    """Call a tool and normalize missing-tool and crash cases into ToolResult."""
    tool_fn = registry.get(tool_id)
    if tool_fn is None:
        return ToolResult(
            tool_id=tool_id,
            output="[Tool not available in this environment]",
            error="not_available",
        )

    try:
        result = tool_fn(action)
    except Exception as exc:  # pragma: no cover - defensive wrapper
        return ToolResult(
            tool_id=tool_id,
            output=f"[Error: {exc}]",
            error=str(exc),
        )

    if not isinstance(result, ToolResult):
        return ToolResult(
            tool_id=tool_id,
            output=f"[Tool error: unexpected return type {type(result).__name__}]",
            error="invalid_return_type",
        )

    if result.tool_id != tool_id:
        result = result.model_copy(update={"tool_id": tool_id})
    return result


def catalog_as_dicts(config: EnvConfig | None = None) -> List[dict[str, Any]]:
    """Convenience helper for JSON serialization."""
    return [asdict(spec) for spec in build_tool_catalog(config)]
