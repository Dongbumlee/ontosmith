"""Entity embedding for candidate search.

The ER gate (Jev) is accurate, but candidates must reach it first.
Substring matching misses abbreviations (LHO, HSCA). Embedding-based
search with a large model finds them.

Validated 2026-10-10 (a "just in case" check was right):
- text-embedding-3-large: LHO→Oswald variants, HSCA→House Select Committee
- all-MiniLM-L6-v2 (local): missed both

That comparison is why there is no local embedding path
:
the embedder is the embedder seat's hosted binding, and
a seat whose key cannot be resolved refuses by name
(:func:`default_embedder`).
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

import numpy as np

from . import connections
from .reference_profile import EMBED_MODEL
from .reference_profile import EMBEDDINGS_URL as EMBEDDINGS_URL


class EntityEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, dims) normalized embeddings."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @property
    def usage(self) -> dict:
        """Cumulative API usage: input_tokens / output_tokens
        / cost, same shape as the Jev decision calls."""
        return {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}


class SeatEmbedder(EntityEmbedder):
    """Large embedding model via the embedder seat's
    binding (preset: OpenRouter). The endpoint and key
    resolve through the connection layer; a binding whose
    key variable is unset refuses with BindingKeyError
    naming the seat and the variable."""

    def __init__(self, model: str = EMBED_MODEL):
        self._model = model
        binding = connections.seat("embedder")
        self._binding = binding
        self._key = binding.require_key()
        self._url = binding.embeddings_url
        self._usage = {"input_tokens": 0, "output_tokens": 0,
                       "cost": 0.0}

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def usage(self) -> dict:
        """Cumulative usage across all embed() calls on this
        instance, from the responses' usage blocks.
        ``cost`` is None once any response carried no cost
        and the binding declares no rates — unmetered is
        reported as unknown, never folded to 0.0 (issue
        #41); ``cost_basis`` says which state applies."""
        return dict(self._usage)

    def embed(self, texts: list[str]) -> np.ndarray:
        # Batch in chunks of 100 (API limits). Each response is large
        # (100 x 3072 floats) and the egress proxy occasionally cuts a
        # chunked read mid-stream — http.client.IncompleteRead killed
        # the flagship INDEX stage on 2026-10-13 — so retry each
        # chunk request with backoff instead of failing the whole call.
        out = []
        for i in range(0, len(texts), 100):
            chunk = texts[i:i + 100]
            body = json.dumps(
                {"model": self._model, "input": chunk}).encode()
            result = None
            delay = 2.0
            for attempt in range(5):
                req = urllib.request.Request(
                    self._url, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self._key}"},
                    method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        result = json.loads(resp.read())
                    break
                except (http.client.IncompleteRead, urllib.error.URLError,
                        TimeoutError, ConnectionError, json.JSONDecodeError):
                    if attempt == 4:
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
            norm = connections.normalize_usage(
                result.get("usage"), self._binding)
            self._usage["input_tokens"] += norm["input_tokens"]
            self._usage["output_tokens"] += norm["output_tokens"]
            self._usage["cost"] = connections.add_cost(
                self._usage["cost"], norm["cost"])
            self._usage["cost_basis"] = norm["cost_basis"]
            data = sorted(result["data"], key=lambda x: x["index"])
            # Convert each sub-batch to float32 immediately: accumulating
            # raw Python float lists for a whole corpus (~50k chunks x
            # 3072 dims) balloons to several GB and gets the process
            # OOM-killed mid-stage (flagship INDEX, 2026-10-13).
            out.append(np.array([d["embedding"] for d in data],
                                dtype=np.float32))
            if len(out) % 50 == 0:
                print(f"[embedder] {min(i + 100, len(texts))}/{len(texts)}",
                      flush=True)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        arr = np.vstack(out)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr /= norms
        return arr


#: Backwards-compatible alias: the class was
#: renamed to name the seat seam, not one provider. The
#: alias is retained for one release.
OpenRouterEmbedder = SeatEmbedder


def default_embedder() -> EntityEmbedder:
    """The embedder seat's hosted embedder.

    The binding resolves through the connection layer;
    an unset or empty key variable refuses with
    BindingKeyError naming the seat and the variable
    (raised by the constructor's ``require_key``) —
    whether the binding was user-declared or the shipped
    preset. There is no local fallback: the one local
    model measured against the hosted embedder missed
    candidates it catches (module docstring), so a
    keyless embedder *use* is a named error, never a
    quieter model. Callers that can serve without an
    embedder at all (lexical + graph search) do not call
    this — they use the no-key-returns-None path
    (``pipeline.search.try_embedder``)."""
    return SeatEmbedder()
