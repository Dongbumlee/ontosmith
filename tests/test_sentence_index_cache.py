"""The file-backed sentence universe
(``pipeline.sentence_index.filebacked_universe``).

Pins:
- values are the embedder's own vectors, widened to
  float64 — identical to the in-memory build, element
  for element;
- a first build is written to disk ("cached") and a
  second call over the same texts reuses it
  ("reused") with a single one-text dimension probe —
  the universe itself is never re-embedded;
- any change to the text list changes the fingerprint
  and forces a rebuild — a stale matrix is never
  served;
- a matrix cached under an embedder of a different
  dimension is rebuilt for the bound embedder, never
  reused;
- an unwritable cache directory falls back to the
  in-memory build ("memory") with identical values.
"""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.sentence_index import (
    filebacked_universe,
    sentence_cache_dir,
    universe_fingerprint,
)

TEXTS = [f"Sentence number {i} about topic {i % 3}."
         for i in range(25)]


class _CountingEmbedder:
    DIMS = 8

    def __init__(self):
        self.texts_seen = 0

    def embed(self, texts):
        self.texts_seen += len(texts)
        out = np.zeros((len(texts), self.DIMS))
        for i, t in enumerate(texts):
            h = int(hashlib.sha1(t.encode()).hexdigest(),
                    16)
            out[i, h % self.DIMS] = 1.0
            out[i, (h >> 3) % self.DIMS] += 0.5
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


class _WideEmbedder(_CountingEmbedder):
    """Same deterministic shape, twice the dimension —
    a different embedder as far as a cached matrix is
    concerned."""

    DIMS = 16


class TestFilebackedUniverse(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "cache"

    def tearDown(self):
        self._tmp.cleanup()

    def test_values_match_in_memory_build(self):
        emb = _CountingEmbedder()
        matrix, backing = filebacked_universe(
            emb, TEXTS, self.dir)
        self.assertEqual(backing, "cached")
        direct = np.asarray(
            _CountingEmbedder().embed(TEXTS),
            dtype=np.float64)
        self.assertEqual(matrix.dtype, np.float64)
        self.assertTrue(np.array_equal(
            np.asarray(matrix), direct))

    def test_second_call_reuses_without_spend(self):
        first = _CountingEmbedder()
        filebacked_universe(first, TEXTS, self.dir)
        self.assertEqual(first.texts_seen, len(TEXTS))
        second = _CountingEmbedder()
        matrix, backing = filebacked_universe(
            second, TEXTS, self.dir)
        self.assertEqual(backing, "reused")
        # the only spend is the one-text dimension
        # probe that validates the reuse
        self.assertEqual(second.texts_seen, 1)
        self.assertIsInstance(matrix, np.memmap)

    def test_mismatched_dimension_rebuilds(self):
        filebacked_universe(
            _CountingEmbedder(), TEXTS, self.dir)
        wide = _WideEmbedder()
        matrix, backing = filebacked_universe(
            wide, TEXTS, self.dir)
        self.assertEqual(backing, "cached")
        # the rebuild embeds the universe once, plus
        # the one probe text that detected the mismatch
        self.assertEqual(wide.texts_seen, len(TEXTS) + 1)
        self.assertEqual(matrix.shape,
                         (len(TEXTS), _WideEmbedder.DIMS))
        direct = np.asarray(
            _WideEmbedder().embed(TEXTS),
            dtype=np.float64)
        self.assertTrue(np.array_equal(
            np.asarray(matrix), direct))
        # and the rebuilt matrix is the one now reused
        again = _WideEmbedder()
        _matrix, backing = filebacked_universe(
            again, TEXTS, self.dir)
        self.assertEqual(backing, "reused")
        self.assertEqual(again.texts_seen, 1)

    def test_changed_texts_rebuild(self):
        filebacked_universe(
            _CountingEmbedder(), TEXTS, self.dir)
        changed = [*TEXTS[:-1], "A different ending."]
        emb = _CountingEmbedder()
        _matrix, backing = filebacked_universe(
            emb, changed, self.dir)
        self.assertEqual(backing, "cached")
        self.assertEqual(emb.texts_seen, len(changed))

    def test_unwritable_cache_falls_back_to_memory(self):
        blocker = Path(self._tmp.name) / "a-file"
        blocker.write_text("not a directory")
        emb = _CountingEmbedder()
        matrix, backing = filebacked_universe(
            emb, TEXTS, blocker / "cache")
        self.assertEqual(backing, "memory")
        direct = np.asarray(
            _CountingEmbedder().embed(TEXTS),
            dtype=np.float64)
        self.assertTrue(np.array_equal(matrix, direct))

    def test_fingerprint_is_order_sensitive(self):
        self.assertNotEqual(
            universe_fingerprint(TEXTS),
            universe_fingerprint(list(reversed(TEXTS))))

    def test_cache_dir_beside_store_source(self):
        store_dir = Path(self._tmp.name) / "batch-x"
        store_dir.mkdir()
        self.assertEqual(
            sentence_cache_dir(store_dir),
            store_dir / ".sentence-index-cache")
        records = store_dir / "triples.jsonl"
        records.write_text("")
        self.assertEqual(
            sentence_cache_dir(records),
            store_dir / ".sentence-index-cache")
        self.assertIsNone(sentence_cache_dir(None))


if __name__ == "__main__":
    unittest.main()
