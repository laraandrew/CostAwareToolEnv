from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from env.config import EnvConfig
from env.models import TOOL_IDS, OrchestratorAction, ToolResult


def make_stub_registry():
    def make_tool(tool_id: str):
        def _tool(action: OrchestratorAction) -> ToolResult:
            payload = action.query or action.expression or action.code_snippet or action.answer or ""
            return ToolResult(tool_id=tool_id, output=f"{tool_id}:{payload}".rstrip(":"))

        return _tool

    return {tool_id: make_tool(tool_id) for tool_id in TOOL_IDS}


@pytest.fixture()
def sample_dataset():
    return [
        {"question": "What is 2 + 2?", "answer": "4", "domain": "math"},
        {"question": "What is 3 + 1?", "answer": "4", "domain": "math"},
    ]


@pytest.fixture()
def app_client(sample_dataset):
    cfg = EnvConfig(
        num_questions=2,
        total_budget=5.0,
        max_steps_per_question=4,
        shuffle_questions=False,
        domain_mix={"math": 1.0},
    )
    app = create_app(config=cfg, tools=make_stub_registry(), dataset=sample_dataset)
    return TestClient(app)
