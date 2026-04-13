"""Commit tool — passes the answer through; grading happens in environment.py."""
from __future__ import annotations

from env.models import OrchestratorAction, ToolResult


def commit_tool(action: OrchestratorAction) -> ToolResult:
    answer = (action.answer or "").strip()
    return ToolResult(
        tool_id="commit",
        output=f"Committed answer: {answer[:200]}",
    )
