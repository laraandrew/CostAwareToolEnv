"""Restricted Python code executor.

Runs code in a sandboxed namespace — blocks os/sys/subprocess imports
and captures stdout. Intended for math / algorithmic tasks.
"""
from __future__ import annotations

import io
import sys
import contextlib

from env.models import OrchestratorAction, ToolResult

_BLOCKED_MODULES = frozenset({
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "multiprocessing", "threading",
    "signal", "pty", "fcntl", "resource", "gc", "inspect",
})

_MAX_OUTPUT_CHARS = 2000


class _BlockedImport:
    """Raise on any import of blocked modules."""
    def __init__(self, original_import):
        self._orig = original_import

    def __call__(self, name, *args, **kwargs):
        base = name.split(".")[0]
        if base in _BLOCKED_MODULES:
            raise ImportError(f"Module '{name}' is not allowed in code_executor")
        return self._orig(name, *args, **kwargs)


def code_executor_tool(action: OrchestratorAction) -> ToolResult:
    code = (action.code_snippet or action.query or "").strip()
    if not code:
        return ToolResult(tool_id="code_executor", output="[No code provided]", error="empty")

    stdout_buf = io.StringIO()
    safe_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()  # type: ignore[union-attr]
            if k not in ("open", "exec", "eval", "compile", "__import__")
        } if isinstance(__builtins__, dict) else {
            k: getattr(__builtins__, k) for k in dir(__builtins__)
            if k not in ("open", "exec", "eval", "compile", "__import__")
        },
        "__import__": _BlockedImport(__import__),
        "print": lambda *a, **kw: print(*a, **kw, file=stdout_buf),
    }

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(code, "<code_executor>", "exec"), safe_globals)  # noqa: S102
        output = stdout_buf.getvalue()[:_MAX_OUTPUT_CHARS] or "[Code ran, no output]"
        return ToolResult(tool_id="code_executor", output=output)
    except Exception as exc:
        return ToolResult(
            tool_id="code_executor",
            output=f"[Execution error: {exc}]",
            error=str(exc),
        )
