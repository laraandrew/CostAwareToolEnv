"""LLM reasoning tool — calls Together AI (or falls back gracefully)."""
from __future__ import annotations

import os

from env.models import OrchestratorAction, ToolResult

_DEFAULT_MODEL = "meta-llama/Llama-3-8b-chat-hf"
_MAX_TOKENS = 512


def llm_reason_tool(action: OrchestratorAction) -> ToolResult:
    prompt = (action.query or "").strip()
    if not prompt:
        return ToolResult(tool_id="llm_reason", output="[No prompt provided]", error="empty")

    api_key = os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_KEY")
    if not api_key:
        return ToolResult(
            tool_id="llm_reason",
            output="[LLM reasoning not configured — set TOGETHER_API_KEY]",
            error="no_api_key",
        )

    try:
        import together  # type: ignore
        client = together.Together(api_key=api_key)
        resp = client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=_MAX_TOKENS,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        return ToolResult(tool_id="llm_reason", output=text.strip()[:2000])
    except ImportError:
        return ToolResult(
            tool_id="llm_reason",
            output="[together package not installed — pip install together]",
            error="import_error",
        )
    except Exception as exc:
        return ToolResult(tool_id="llm_reason", output=f"[LLM error: {exc}]", error=str(exc))
