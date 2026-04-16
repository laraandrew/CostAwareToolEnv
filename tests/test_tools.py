from __future__ import annotations

from urllib.error import HTTPError

from env.models import OrchestratorAction
from tools.calculator import calculator_tool
from tools.code_executor import code_executor_tool
from tools.commit import commit_tool
from tools.llm_reason import llm_reason_tool
from tools.ceramic_search import make_search_tool
from tools.wiki_lookup import wiki_lookup_tool


def test_calculator_happy_path():
    result = calculator_tool(OrchestratorAction(tool_id="calculator", expression="2 + 2 * 3"))
    assert result.output == "8"
    assert result.error is None


def test_calculator_invalid_input():
    result = calculator_tool(OrchestratorAction(tool_id="calculator", expression="open('x')"))
    assert result.error is not None
    assert result.output.startswith("[Calc error:")


def test_search_fallback_is_deterministic(monkeypatch):
    monkeypatch.delenv("CERAMIC_API_KEY", raising=False)
    monkeypatch.delenv("SEE_CERAMIC_API_KEY", raising=False)

    tool = make_search_tool()
    action = OrchestratorAction(tool_id="ceramic_search", query="Eiffel Tower")

    first = tool(action)
    second = tool(action)

    assert first.error is None
    assert first.output == second.output
    assert "Eiffel Tower" in first.output


def test_wiki_lookup_not_found(monkeypatch):
    def fake_urlopen(*args, **kwargs):
        raise HTTPError(url="https://example.com", code=404, msg="not found", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = wiki_lookup_tool(OrchestratorAction(tool_id="wiki_lookup", query="Definitely Not A Real Page"))
    assert result.error == "not_found"
    assert "no article found" in result.output.lower()


def test_llm_reason_no_api_key(monkeypatch):
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_KEY", raising=False)

    result = llm_reason_tool(OrchestratorAction(tool_id="llm_reason", query="Explain gravity"))
    assert result.error == "no_api_key"
    assert "not configured" in result.output.lower()


def test_commit_passthrough_behavior():
    result = commit_tool(OrchestratorAction(tool_id="commit", answer="  1889  "))
    assert result.tool_id == "commit"
    assert result.output == "Committed answer: 1889"
