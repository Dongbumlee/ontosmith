"""The build-time sentence universe — one definition,
shared by every surface that touches sentence serving.

Sentence serving (``pipeline.search.sentence_search``)
retrieves over a direct index of EVERY sentence in the
build's chunks, not just the sentences of whichever
chunks a chunk arm happened to surface. That universe
has exactly one definition, fixed by the serving
measurements it reproduces:

- documents in sorted order, chunks in the chunker's
  order (the same iteration the chunk index itself is
  built in);
- each chunk's text split with the proposer's
  deterministic sentence splitter;
- deduplicated by whitespace-normalized text — the
  first occurrence wins and lends the sentence its
  (doc_id, chunk_id) anchor.

The INDEX stage builds the rows with this function and
embeds them with the build's embedder (the same seat
that embedded the chunks); the serving stores build
the identical rows at query time when a build predates
the shipped sentence index. One builder, so the two
paths cannot drift apart.
"""
from __future__ import annotations

import re


def ws_norm(s: str) -> str:
    """Whitespace-collapsed, lowercased text identity —
    the dedupe and coverage key for sentence units."""
    return " ".join((s or "").split()).lower()


def build_sentence_rows(chunks: list[str],
                        meta: list[dict]) -> list[dict]:
    """The sentence universe over a chunk list.

    ``chunks`` / ``meta`` are the chunk index's own pair
    (meta rows carry doc_id + chunk_idx + batch_id, the
    tier2_index.pkl shape). Returns rows
    ``{doc_id, chunk_id, text}`` in universe order;
    chunk ids take the store's deterministic form
    (``pipeline.graph_store.chunk_id_for`` — imported
    lazily so this module stays importable from the
    serving layer without a cycle)."""
    from .graph_store import chunk_id_for
    from .proposer import split_sentences

    rows: list[dict] = []
    seen: set[str] = set()
    for text, m in zip(chunks, meta, strict=True):
        cid = chunk_id_for(m.get("batch_id", ""),
                           m["doc_id"], int(m["chunk_idx"]))
        for s in split_sentences(text):
            key = ws_norm(s)
            if key and key not in seen:
                seen.add(key)
                rows.append({"doc_id": m["doc_id"],
                             "chunk_id": cid, "text": s})
    return rows


def ent_key(name: str) -> str:
    """Entity-surface identity for the beam channel's
    adjacency: whitespace-collapsed, lowercased surface
    name (the measurement harness's key, verbatim)."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())
