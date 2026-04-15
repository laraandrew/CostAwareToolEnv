"""Wikipedia lookup tool — returns the intro paragraph of an article."""
from __future__ import annotations

import urllib.parse
import urllib.request
import json

from env.models import OrchestratorAction, ToolResult

_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def wiki_lookup_tool(action: OrchestratorAction) -> ToolResult:
    query = (action.query or "").strip()
    if not query:
        return ToolResult(tool_id="wiki_lookup", output="[No query provided]", error="empty_query")

    title = urllib.parse.quote(query.replace(" ", "_"))
    url = _WIKI_API.format(title)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CostAwareToolEnv/0.1"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        extract = data.get("extract", "").strip()
        page_title = data.get("title", query)
        if not extract:
            return ToolResult(tool_id="wiki_lookup", output=f"[No summary found for '{query}']")
        return ToolResult(tool_id="wiki_lookup", output=f"**{page_title}**\n{extract[:800]}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return ToolResult(
                tool_id="wiki_lookup",
                output=f"[Wikipedia: no article found for '{query}']",
                error="not_found",
            )
        return ToolResult(tool_id="wiki_lookup", output=f"[Wiki HTTP error {exc.code}]", error=str(exc))
    except Exception as exc:
        return ToolResult(tool_id="wiki_lookup", output=f"[Wiki error: {exc}]", error=str(exc))
