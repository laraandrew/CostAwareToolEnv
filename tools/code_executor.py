"""Restricted Python code executor.

The executor is intentionally narrow:
  - import statements are rejected before execution
  - dangerous builtin names are blocked
  - dunder attribute access is blocked to prevent object graph escapes
  - only a curated builtin/module surface is exposed

This keeps the tool useful for HumanEval-style tasks while making the
security boundaries explicit and testable.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import math
import operator
import collections
import functools
import itertools
import statistics
import heapq
import bisect
import fractions
import decimal
import re
from typing import Any, Dict

from env.models import OrchestratorAction, ToolResult

_MAX_OUTPUT_CHARS = 2000

_BLOCKED_NAMES = {
    "__builtins__",
    "__import__",
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "input",
    "help",
    "type",
    "object",
    "super",
    "memoryview",
    "breakpoint",
    "exit",
    "quit",
}

_BLOCKED_ATTRS = {
    "__class__",
    "__base__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__dict__",
    "__getattribute__",
    "__getattr__",
    "__setattr__",
    "__delattr__",
    "__reduce__",
    "__reduce_ex__",
    "__func__",
    "__self__",
    "__module__",
}

_SAFE_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "chr",
    "dict",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    "divmod",
    "ord",
    "Exception",
    "ValueError",
    "RuntimeError",
    "TypeError",
    "KeyError",
    "IndexError",
    "AssertionError",
    "ZeroDivisionError",
    "object",
}

_SAFE_MODULES: Dict[str, Any] = {
    "math": math,
    "collections": collections,
    "functools": functools,
    "itertools": itertools,
    "statistics": statistics,
    "heapq": heapq,
    "bisect": bisect,
    "fractions": fractions,
    "decimal": decimal,
    "re": re,
    "operator": operator,
}


class SandboxViolation(ValueError):
    """Raised when code tries to cross the sandbox boundary."""


def _validate_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation("import statements are blocked")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SandboxViolation("global and nonlocal are blocked")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _BLOCKED_ATTRS:
                raise SandboxViolation(f"attribute access '{node.attr}' is blocked")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _BLOCKED_NAMES:
                raise SandboxViolation(f"name '{node.id}' is blocked")


def _safe_builtins(stdout_buf: io.StringIO) -> Dict[str, Any]:
    safe: Dict[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}

    def safe_print(*args: Any, **kwargs: Any) -> None:
        kwargs = dict(kwargs)
        kwargs.pop("file", None)
        builtins.print(*args, **kwargs, file=stdout_buf)

    safe["print"] = safe_print
    safe["__build_class__"] = builtins.__build_class__
    return safe


def code_executor_tool(action: OrchestratorAction) -> ToolResult:
    code = (action.code_snippet or action.query or "").strip()
    if not code:
        return ToolResult(tool_id="code_executor", output="[No code provided]", error="empty")

    stdout_buf = io.StringIO()
    safe_globals: Dict[str, Any] = {
        "__builtins__": _safe_builtins(stdout_buf),
        "__name__": "__code_executor__",
        "__package__": None,
        **_SAFE_MODULES,
    }

    try:
        tree = ast.parse(code, mode="exec")
        _validate_tree(tree)
        with contextlib.redirect_stdout(stdout_buf):
            exec(compile(tree, "<code_executor>", "exec"), safe_globals)  # noqa: S102
        output = stdout_buf.getvalue()[:_MAX_OUTPUT_CHARS] or "[Code ran, no output]"
        return ToolResult(tool_id="code_executor", output=output)
    except SandboxViolation as exc:
        return ToolResult(tool_id="code_executor", output=f"[Sandbox blocked: {exc}]", error="sandbox_violation")
    except Exception as exc:
        return ToolResult(tool_id="code_executor", output=f"[Execution error: {exc}]", error=str(exc))
