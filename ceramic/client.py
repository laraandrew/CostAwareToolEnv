"""Ceramic AI search client.

Matches the interface used by SearchEconomicsEnv so both environments
share the same retrieval backend.

API key priority:
  1. CERAMIC_API_KEY  env var
  2. SEE_CERAMIC_API_KEY env var  (HF Spaces compatibility with SearchEcon)
  3. Falls back to FallbackCeramicClient  (offline / CI, fully deterministic)

Ceramic API notes (verified 2025):
  - Endpoint : POST https://api.ceramic.ai/search
  - Body     : {"query": "<string>"}   (no pagination params supported)
  - Response : {"requestId": "...", "result": {"results": [...], "totalResults": N}}
  - Each result has: title, url, description, score
  - Always returns up to 10 results per call
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import List

import httpx


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------

class CeramicClient:
    """Thin wrapper around the Ceramic search API."""

    BASE_URL = "https://api.ceramic.ai"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search Ceramic and return up to top_k results (max 10)."""
        if not query.strip():
            return []
        resp = self._client.post(
            f"{self.BASE_URL}/search",
            json={"query": query},
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("result", {}).get("results", [])
        results = []
        for item in raw[:top_k]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                description=item.get("description", ""),
                score=float(item.get("score", 0.0)),
            ))
        return results

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Offline fallback
# ---------------------------------------------------------------------------

class FallbackCeramicClient:
    """Deterministic offline client used when no API key is available.

    Generates reproducible fake results via SHA-256 hashing so tests
    and CI runs are stable without network access.
    """

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        h = int(hashlib.sha256(query.encode()).hexdigest(), 16)
        results = []
        for i in range(min(top_k, 3)):
            seed = (h + i) % 10_000
            results.append(SearchResult(
                title=f"Result {seed}: {query[:40]}",
                url=f"https://fallback.example.com/doc/{seed}",
                description=f"Offline fallback result #{i+1} for query: {query}",
                score=round(0.9 - i * 0.15, 3),
            ))
        return results

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_ceramic_client() -> CeramicClient | FallbackCeramicClient:
    """Return a live CeramicClient if a key is set, otherwise FallbackCeramicClient."""
    key = (
        os.environ.get("CERAMIC_API_KEY")
        or os.environ.get("SEE_CERAMIC_API_KEY")
    )
    if key:
        return CeramicClient(api_key=key)
    return FallbackCeramicClient()
