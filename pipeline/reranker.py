"""The rerank step for sentence serving.

The serving measurements' final form reranks the merged
head (top 80 units) with a local cross-encoder before
the budget fill — the step that separated S3 from S2.
The reference implementation is that cross-encoder,
loaded lazily in-process: it is a local model, not a
provider seat, so there is no key and no connection
binding; when the optional ``rerank`` extra
(sentence-transformers) is not installed, :func:`try_reranker`
returns None and sentence serving fills in merge order
and says so in its notes — the step is toggleable, never
a silent no-op.
"""
from __future__ import annotations

from .reference_profile import RERANK_MODEL


class LocalCrossEncoderReranker:
    """bge-reranker-base (the reference profile's rerank
    binding) over (query, text) pairs. ``score`` returns
    one float per text, higher = more relevant."""

    def __init__(self, model: str = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model)
        self.model_name = model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


def try_reranker() -> LocalCrossEncoderReranker | None:
    """The reference reranker when its extra is
    installed; None otherwise (the caller notes the
    inactive step). A model that fails to LOAD is also
    None — a broken local install must not take the
    whole search surface down with it."""
    try:
        return LocalCrossEncoderReranker()
    except Exception:
        return None
