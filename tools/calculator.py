"""Safe AST-based calculator tool.

Supports arithmetic, comparisons, and basic math functions.
No exec/eval with arbitrary code — uses ast.literal_eval-style restricted eval.
"""
from __future__ import annotations

import ast
import math
import operator
from typing import Any

from env.models import OrchestratorAction, ToolResult

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil,
    "pi": math.pi, "e": math.e,
}


def _safe_eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            return _SAFE_FUNCS[node.id]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Compare):
        left = _safe_eval(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _SAFE_COMPARISONS:
                raise ValueError(f"Unsupported comparison: {op_type.__name__}")
            right = _safe_eval(comparator)
            if not _SAFE_COMPARISONS[op_type](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        if not callable(func):
            raise ValueError("Not callable")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator_tool(action: OrchestratorAction) -> ToolResult:
    expr = (action.expression or action.query or "").strip()
    if not expr:
        return ToolResult(tool_id="calculator", output="[No expression provided]", error="empty")
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree)
        return ToolResult(tool_id="calculator", output=str(result))
    except Exception as exc:
        return ToolResult(tool_id="calculator", output=f"[Calc error: {exc}]", error=str(exc))
