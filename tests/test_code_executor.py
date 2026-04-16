from __future__ import annotations

from env.models import OrchestratorAction
from tools.code_executor import code_executor_tool


def test_code_executor_empty_input():
    result = code_executor_tool(OrchestratorAction(tool_id="code_executor"))
    assert result.error == "empty"
    assert "No code provided" in result.output


def test_code_executor_runtime_error():
    result = code_executor_tool(
        OrchestratorAction(tool_id="code_executor", code_snippet="print(1 / 0)")
    )
    assert result.error is not None
    assert "division by zero" in result.output.lower()


def test_code_executor_blocks_imports():
    result = code_executor_tool(
        OrchestratorAction(tool_id="code_executor", code_snippet="import os\nprint('hi')")
    )
    assert result.error == "sandbox_violation"
    assert "import statements are blocked" in result.output


def test_code_executor_blocks_unsafe_builtins():
    result = code_executor_tool(
        OrchestratorAction(tool_id="code_executor", code_snippet="open('tmp.txt', 'w')")
    )
    assert result.error == "sandbox_violation"
    assert "name 'open' is blocked" in result.output


def test_code_executor_blocks_escape_attempts():
    result = code_executor_tool(
        OrchestratorAction(
            tool_id="code_executor",
            code_snippet="().__class__.__mro__[1].__subclasses__()",
        )
    )
    assert result.error == "sandbox_violation"
    assert "__class__" in result.output or "__subclasses__" in result.output
