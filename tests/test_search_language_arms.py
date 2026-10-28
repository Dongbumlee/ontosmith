"""Search language-arm tests (issue #53: B1–B3).

The pilot's code findings: the query tokenizer was
``[a-z0-9]+``, so a Korean-only query produced ZERO tokens
— the lexical arm never fired (B1); the graph arm's overlap
sets are built from the same tokenizer, so Korean triples
were invisible to it (B2); and the fulltext indexes were
created with no analyzer option, leaving non-Latin text on
the stock analyzer with no supported way to choose another
(B3).

Pinned here:
- B1: tokenize is script-aware AND byte-identical on ASCII
  (the flagship mirror is an English claim; it must not
  move). Korean queries produce tokens; the lexical scorer
  matches Korean chunk text.
- B2: the graph overlap ranking ranks a Korean document
  for a Korean query, and hybrid_search over a Korean
  fixture store fires BOTH the lexical and graph arms.
- B3: the analyzer knob — unset means the pre-fix schema
  statements byte-for-byte; set means both fulltext
  indexes carry it; a malformed name fails closed.
"""
import os
import re
import unittest
from unittest import mock

from pipeline import graph_store, query_stack, search

_OLD = re.compile(r"[a-z0-9]+")


class TestTokenizeScriptAware(unittest.TestCase):
    def test_ascii_stream_unchanged(self):
        samples = [
            "What contracts govern the Alder & Finch merger?",
            "Section 10.2(b) — termination_for convenience, 30 days",
            "Unicode-free plain ASCII only 12345",
            "a_b c__d e-f g.h",
            "",
        ]
        for s in samples:
            self.assertEqual(query_stack.tokenize(s),
                             _OLD.findall(s.lower()), s)

    def test_korean_query_produces_tokens(self):
        toks = query_stack.tokenize("삼성전자 본사 위치")
        self.assertEqual(toks, ["삼성전자", "본사", "위치"])

    def test_mixed_and_cjk(self):
        self.assertEqual(
            query_stack.tokenize("Samsung 삼성전자와 Sony"),
            ["samsung", "삼성전자와", "sony"])
        self.assertTrue(query_stack.tokenize("東京 の 天気"))

    def test_none_is_empty(self):
        self.assertEqual(query_stack.tokenize(None), [])


class TestLexicalArmKorean(unittest.TestCase):
    def test_lexical_score_matches_korean(self):
        terms = search._query_terms("삼성전자 본사")
        self.assertTrue(terms)
        score = search._lexical_score(
            "삼성전자 본사 는 경기도 수원시 에 있다", terms)
        self.assertGreater(score, 0.0)


class TestGraphArmKorean(unittest.TestCase):
    def test_overlap_ranking_korean(self):
        rows = [{"doc_id": "doc-ko", "p_true": 0.9, "tier": 1},
                {"doc_id": "doc-en", "p_true": 0.9, "tier": 1}]
        texts = ["삼성전자 본사 위치 경기도 수원시",
                 "Samsung headquarters located Suwon"]
        token_sets = [set(query_stack.tokenize(t)) for t in texts]
        ranked = query_stack._overlap_ranking(
            rows, token_sets, "삼성전자 본사 위치")
        self.assertEqual(ranked, ["doc-ko"])


class _KoreanStore:
    kind = "double"

    def __init__(self):
        self._triples = [{
            "subject_id": "e1", "subject": "삼성전자",
            "relation": "본사 위치", "object_id": "e2",
            "object": "경기도 수원시", "doc_id": "doc-ko",
            "quote": "삼성전자 본사", "p_true": 0.9, "tier": 1,
            "tag": "VERIFIED", "batch_id": "fx", "model": "",
            "gate": ""}]
        self._chunks = [{
            "chunk_id": "fx:doc-ko:0", "doc_id": "doc-ko",
            "idx": 0, "tier": 1, "tag": "VERIFIED",
            "text": "삼성전자 본사 는 경기도 수원시 에 있다. "
                    "삼성전자 는 전자 제품 을 만든다."}]

    def describe(self):
        return {"kind": "double", "batch_id": "fx",
                "entities": 2, "triples": 1, "documents": 1,
                "chunks": 1, "embeddings": False}

    def batch_filter(self):
        return {}

    def entities(self):
        return {"삼성전자": "e1", "경기도 수원시": "e2"}

    def entity(self, entity_id):
        names = {"e1": "삼성전자", "e2": "경기도 수원시"}
        if entity_id in names:
            return {"entity_id": entity_id,
                    "name": names[entity_id], "aliases": []}
        return None

    def find_entities(self, query, limit=10):
        return [{"entity_id": "e1", "name": "삼성전자",
                 "aliases": []}] if "삼성전자" in query else []

    def triples(self):
        return [dict(t) for t in self._triples]

    def triple_records(self):
        return [dict(t) for t in self._triples]

    def neighborhood(self, entity_ids, hops=1, tier_filter=None,
                     limit=200):
        return {"nodes": [], "edges": []}

    def search_chunks(self, vector=None, text=None, filters=None,
                      k=10):
        if vector is not None:
            return []
        terms = {t for t in query_stack.tokenize(text or "")
                 if len(t) > 2}
        out = []
        for c in self._chunks:
            matched = terms & set(query_stack.tokenize(c["text"]))
            if matched:
                out.append({**c, "score": float(len(matched)),
                            "text_score": float(len(matched))})
        return out[:k]


class TestHybridSearchKorean(unittest.TestCase):
    def test_all_arms_fire_for_korean_query(self):
        result = search.hybrid_search(_KoreanStore(),
                                      "삼성전자 본사 위치")
        self.assertIn("lexical", result["arms"])
        self.assertIn("graph", result["arms"])
        doc_ids = [d["doc_id"] for d in result["documents"]]
        self.assertIn("doc-ko", doc_ids)
        self.assertTrue(result["triples"])


class TestFulltextAnalyzerKnob(unittest.TestCase):
    def test_default_statements_unchanged(self):
        self.assertEqual(graph_store.schema_statements(),
                         graph_store.SCHEMA_STATEMENTS)
        joined = " ".join(graph_store.SCHEMA_STATEMENTS)
        self.assertNotIn("fulltext.analyzer", joined)

    def test_analyzer_applies_to_both_fulltext_indexes(self):
        stmts = graph_store.schema_statements("cjk")
        ft = [s for s in stmts if "FULLTEXT" in s]
        self.assertEqual(len(ft), 2)
        for s in ft:
            self.assertIn("`fulltext.analyzer`: 'cjk'", s)

    def test_env_knob(self):
        with mock.patch.dict(
                os.environ,
                {graph_store.FULLTEXT_ANALYZER_ENV: "cjk"}):
            self.assertEqual(graph_store.fulltext_analyzer(),
                             "cjk")
        with mock.patch.dict(
                os.environ,
                {graph_store.FULLTEXT_ANALYZER_ENV: "  "}):
            self.assertIsNone(graph_store.fulltext_analyzer())

    def test_malformed_name_fails_closed(self):
        with self.assertRaises(ValueError):
            graph_store.schema_statements("cjk'} DROP")


if __name__ == "__main__":
    unittest.main()
