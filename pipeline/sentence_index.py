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

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np


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


def universe_fingerprint(texts: list[str]) -> str:
    """Content identity of an ordered sentence list —
    the cache key for a query-time-built universe. Any
    change to any text, or to the order, changes the
    fingerprint, so a stale matrix is never reused."""
    h = hashlib.sha256()
    for t in texts:
        b = t.encode()
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


#: Sub-batch size for a file-backed universe build:
#: each sub-batch's vectors are flushed to disk before
#: the next is embedded, so an interrupted build loses
#: at most one sub-batch of spend on the next attempt
#: (the partial file is never reused — only a complete,
#: marker-stamped matrix is).
_UNIVERSE_BATCH = 500


def sentence_cache_dir(source) -> Path | None:
    """Where a store keeps its query-time-built
    sentence matrix: a cache directory beside the
    store's own artifacts (inside a batch directory,
    or next to a records file), so the matrix lives
    and dies with the store it serves."""
    if source is None:
        return None
    src = Path(source)
    base = src if src.is_dir() else src.parent
    return base / ".sentence-index-cache"


def filebacked_universe(
        embedder, texts: list[str],
        cache_dir: Path | None) -> tuple[np.ndarray, str]:
    """Embed ``texts`` as the serving matrix (float64)
    and return ``(matrix, backing)`` — backing is
    ``"reused"`` (a complete cached matrix was already
    on disk), ``"cached"`` (embedded now and written to
    disk), or ``"memory"`` (no writable cache directory;
    the plain in-memory build).

    Why file-backed: materializing a large universe in
    anonymous memory peaks at the float32 matrix plus
    its float64 widening at once (~4.7 GB at 127k
    sentences x 3072 dims) — enough to kill serving on
    a modest machine at exactly the moment a build that
    predates the shipped index is first searched. A
    file-backed matrix holds identical values (the
    embedder's own vectors, widened) while its pages
    stay evictable, and a later process reuses the file
    instead of paying the embedding spend again."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float64), \
            "memory"
    if cache_dir is not None:
        try:
            return _filebacked(embedder, texts,
                               Path(cache_dir))
        except OSError:
            pass  # unwritable cache: in-memory fallback
    return (np.asarray(embedder.embed(texts),
                       dtype=np.float64), "memory")


def _filebacked(embedder, texts: list[str],
                cache_dir: Path) -> tuple[np.ndarray,
                                           str]:
    fp = universe_fingerprint(texts)
    stem = f"universe-{fp[:16]}"
    final = cache_dir / f"{stem}.npy"
    marker = cache_dir / f"{stem}.json"
    if final.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get("fingerprint") == fp and \
                saved.get("count") == len(texts):
            mm = np.load(final, mmap_mode="r")
            if mm.shape == (len(texts),
                            int(saved["dims"])) and \
                    mm.dtype == np.float64:
                return mm, "reused"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=cache_dir, prefix=f"{stem}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        mm = None
        dims = 0
        for i in range(0, len(texts), _UNIVERSE_BATCH):
            vecs = np.asarray(
                embedder.embed(texts[i:i + _UNIVERSE_BATCH]))
            if mm is None:
                dims = int(vecs.shape[1])
                mm = np.lib.format.open_memmap(
                    tmp, mode="w+", dtype=np.float64,
                    shape=(len(texts), dims))
            mm[i:i + len(vecs)] = vecs
            mm.flush()
        assert mm is not None
        del mm
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    mtmp = marker.with_suffix(".json.tmp")
    mtmp.write_text(json.dumps(
        {"fingerprint": fp, "count": len(texts),
         "dims": dims}))
    os.replace(mtmp, marker)
    return np.load(final, mmap_mode="r"), "cached"


def ent_key(name: str) -> str:
    """Entity-surface identity for the beam channel's
    adjacency: whitespace-collapsed, lowercased surface
    name (the measurement harness's key, verbatim)."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())
