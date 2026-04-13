"""Compatibility shim — real code lives in ceramic/client.py.

Mirrors the SearchEconomicsEnv CeramicClient interface so the two
environments share the same retrieval backend.

Priority for the API key:
  1. CERAMIC_API_KEY env var
  2. SEE_CERAMIC_API_KEY env var (HF Spaces compatibility)
  3. Falls back to FallbackCeramicClient (offline, deterministic)
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

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

    BASE_URL = "https://api.ceramic.ai/v1"

    def __init__(self, api_key: str):
        self._key = api_key
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10.0,
        )

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        resp = self._client.post(
            f"{self.BASE_URL}/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("results", []):
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
    """Deterministic offline client — used when no API key is set."""

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Stable hash → reproducible fake results per query
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

_DEFAULT_KEY = "cer_sk_live_543fe74e79df_eyJvcmdfaWQiOiJvcmdfMDFLTlpINkU5RVNDTUowUUoyREpINFZWWEYiLCJrZXlfaWQiOiI1NDNmZTc0ZTc5ZGYifQ.k8I4Aljsk29y4Uki37Wxfd7QZHs40XSJVNBNnfksCtM"


def get_ceramic_client() -> CeramicClient | FallbackCeramicClient:
    key = (
        os.environ.get("CERAMIC_API_KEY")
        or os.environ.get("SEE_CERAMIC_API_KEY")
        or _DEFAULT_KEY
    )
    if key:
        return CeramicClient(api_key=key)
    return FallbackCeramicClient()
