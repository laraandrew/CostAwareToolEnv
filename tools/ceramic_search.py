"""Ceramic search tool — wraps CeramicClient."""
from __future__ import annotations

from ceramic.client import get_ceramic_client
from env.models import OrchestratorAction, ToolResult


def make_search_tool(top_k: int = 3):
    """Factory: creates a search tool with a shared Ceramic client."""
    client = get_ceramic_client()

    def _search(action: OrchestratorAction) -> ToolResult:
        query = (action.query or "").strip()
        if not query:
            return ToolResult(
                tool_id="ceramic_search",
                output="[No query provided]",
                error="empty_query",
            )
        try:
            results = client.search(query, top_k=top_k)
            snippets = []
            for r in results:
                snippets.append(f"**{r.title}** ({r.score:.2f})\n{r.description}")
            output = "\n\n".join(snippets) if snippets else "[No results found]"
            return ToolResult(tool_id="ceramic_search", output=output)
        except Exception as exc:
            return ToolResult(
                tool_id="ceramic_search",
                output=f"[Search error: {exc}]",
                error=str(exc),
            )

    return _search
