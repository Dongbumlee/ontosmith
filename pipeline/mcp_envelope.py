"""The MCP result envelope — the shape every MCP tool
returns (design: mcp-interface-design.md §2/§7).

One envelope for all tools::

    {"docs": {"d1": {"doc_id": ...}},
     "items": [...], "returned": n, "total": m,
     "truncated": bool, "cursor": "..."}   # cursor only
                                           # when truncated

Rules implemented here, once, for every tool:

- **Hoisted documents.** Source-document metadata appears
  once in ``docs`` under a short ref (``d1``); items cite
  the ref instead of repeating the document identity.
- **Citation-ready keys.** Every item carries ``cite`` —
  ``<doc-ref>:s<n>``, the n-th cited item for that
  document in this response — so a caller's prose can
  cite in the same keys.
- **Hard budget, signalled truncation.** ``max_tokens``
  is a ceiling on the item payload, estimated
  deterministically (chars / 4, the standard rough
  token proxy). When the budget bites, items are cut at
  an item boundary, ``truncated`` is true, and ``cursor``
  carries the offset to resume from — never a silent cut.
- **Stable triple ids.** A triple's id is a hash of its
  (subject_id, relation, object_id) key: composable
  across calls and stores, derivable by any holder of
  the key, no server state.

Pure functions and one small builder class; no store,
no transport. The service layer (pipeline.mcp_server)
composes items and hands them here for budgeting.
"""
from __future__ import annotations

import base64
import hashlib
import json

#: Rough token proxy: one token ≈ four characters of
#: JSON. Deliberately conservative and deterministic —
#: the budget is a ceiling, not a measurement.
CHARS_PER_TOKEN = 4


def estimate_tokens(payload) -> int:
    """The deterministic token estimate of a JSON-able
    payload, under the envelope's chars/4 proxy."""
    return len(json.dumps(payload, ensure_ascii=False)) \
        // CHARS_PER_TOKEN + 1


def triple_id(subject_id: str, relation: str,
              object_id: str) -> str:
    """The stable id of one triple: ``t_`` + the first
    12 hex chars of the key's SHA-1. Two records with the
    same (subject, relation, object) key are attestations
    of the same triple and share its id."""
    key = "\x1f".join((subject_id, relation, object_id))
    return "t_" + hashlib.sha1(
        key.encode("utf-8")).hexdigest()[:12]


def encode_cursor(offset: int) -> str:
    """An opaque resume token: the next item offset,
    base64-wrapped so callers treat it as a string to
    hand back, not a number to arithmetic on."""
    return base64.urlsafe_b64encode(
        str(offset).encode("ascii")).decode("ascii")


def decode_cursor(cursor: str | None) -> int:
    """The offset a cursor carries; 0 for no cursor.
    A malformed cursor is a corrective refusal (the
    caller can learn from it), never a silent restart."""
    if not cursor:
        return 0
    try:
        return int(base64.urlsafe_b64decode(
            cursor.encode("ascii")).decode("ascii"))
    except Exception:
        raise ValueError(
            f"invalid cursor {cursor!r}: pass back the "
            "exact cursor string a previous response "
            "returned, or omit it to start over") from None


class EnvelopeBuilder:
    """Accumulates items against one response's document
    registry, then renders the envelope under a token
    budget.

    Usage: ``add`` each item (a dict whose evidence/text
    entries reference documents by ``doc_id`` via the
    ``doc`` field the service sets through
    :meth:`doc_ref`), then :meth:`build` with the full
    item list's total and the resume offset.
    """

    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self._refs: dict[str, str] = {}
        self._cite_counts: dict[str, int] = {}

    def doc_ref(self, doc_id: str) -> str:
        """The short ref for a source document,
        registering it on first sight (``d1``, ``d2``,
        ... in encounter order)."""
        ref = self._refs.get(doc_id)
        if ref is None:
            ref = f"d{len(self._refs) + 1}"
            self._refs[doc_id] = ref
            self.docs[ref] = {"doc_id": doc_id}
        return ref

    def cite_key(self, doc_ref: str) -> str:
        """The next citation key for a document ref:
        ``d1:s1``, ``d1:s2``, ... in item order."""
        n = self._cite_counts.get(doc_ref, 0) + 1
        self._cite_counts[doc_ref] = n
        return f"{doc_ref}:s{n}"

    def build(self, items: list[dict], *, total: int,
              offset: int = 0,
              max_tokens: int = 2000) -> dict:
        """Render the envelope: the items (already
        windowed by the caller to start at ``offset``)
        cut at an item boundary when their estimated
        size exceeds ``max_tokens``. ``total`` is the
        full result count before windowing, so the
        caller can report coverage honestly."""
        kept: list[dict] = []
        truncated = False
        for item in items:
            trial = [*kept, item]
            if kept and estimate_tokens(trial) > max_tokens:
                truncated = True
                break
            kept = trial
        else:
            truncated = offset + len(items) < total
        out: dict = {
            "docs": self.docs,
            "items": kept,
            "returned": len(kept),
            "total": total,
            "truncated": truncated,
        }
        if truncated:
            out["cursor"] = encode_cursor(
                offset + len(kept))
        return out
