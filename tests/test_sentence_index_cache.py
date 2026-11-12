"""The file-backed sentence universe
(``pipeline.sentence_index.filebacked_universe``).

Pins:
- values are the embedder's own vectors, widened to
  float64 — identical to the in-memory build, element
  for element;
- a first build is written to disk ("cached") and a
  second call over the same texts reuses it
  ("reused") with zero embedder calls;
- any change to the text list changes the fingerprint
  and forces a rebuild — a stale matrix is never
  served;
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
        self.assertEqual(second.texts_seen, 0)
        self.assertIsInstance(matrix, np.memmap)

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
