"""Tool registry for CostAwareToolEnv.

Each tool is a callable: (action: OrchestratorAction) -> ToolResult
"""
from __future__ import annotations

from typing import Callable, Dict

from env.config import EnvConfig
from env.models import OrchestratorAction, ToolResult

from .calculator import calculator_tool
from .ceramic_search import make_search_tool
from .code_executor import code_executor_tool
from .commit import commit_tool
from .llm_reason import llm_reason_tool
from .wiki_lookup import wiki_lookup_tool


def build_tool_registry(config: EnvConfig | None = None) -> Dict[str, Callable]:
    """Return a mapping of tool_id → tool function."""
    return {
        "ceramic_search": make_search_tool(),
        "calculator":     calculator_tool,
        "wiki_lookup":    wiki_lookup_tool,
        "code_executor":  code_executor_tool,
        "llm_reason":     llm_reason_tool,
        "commit":         commit_tool,
    }
