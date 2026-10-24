"""Canonicalization placement.

The module is a measurement instrument and lives in the
harness; it was pipeline/canonicalization.py
under an experiment quarantine. The golden
(tests/golden/canonicalization_fixture.json) was captured
from the PRE-MOVE module on the fixture below — the moved
module must reproduce it byte-for-byte: same enriched
roster, same stats, same candidates, same pack-data
qualifier set.
"""
import importlib.util
import json
import unittest
from pathlib import Path

from harness import canonicalization

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads(
    (ROOT / "tests" / "golden" / "canonicalization_fixture.json")
    .read_text())

TEXT = (
    "acme corp (ac) is here. acme corp grew. ac fell. "
    "smith/jones signed the deal. smith left. jones stayed. "
    "delta air lines flies daily. delta air is a brand. "
    "globex is a rival. globex corp (globex) hired staff. "
    "red cross society met. red cross foundation met. "
    "red cross helped. "
    "brca1 gene mutation causes disease. brca1 was studied."
)
MENTIONS = [
    {"surface": "acme corp (ac)", "type": "ORG"},
    {"surface": "smith/jones", "type": "ORG"},
    {"surface": "delta air lines", "type": "ORG"},
    {"surface": "globex", "type": "ORG"},
    {"surface": "globex corp (globex)", "type": "ORG"},
    {"surface": "red cross society", "type": "ORG"},
    {"surface": "red cross foundation", "type": "ORG"},
    {"surface": "brca1", "type": "Gene"},
]


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class TestCanonicalizationGolden(unittest.TestCase):
    def test_fixture_byte_identical(self):
        enriched, stats = canonicalization.canonicalize_doc(
            MENTIONS, TEXT)
        self.assertEqual(_canon(enriched),
                         _canon(GOLDEN["enriched"]))
        self.assertEqual(_canon(stats), _canon(GOLDEN["stats"]))

    def test_candidates_byte_identical(self):
        cands = canonicalization.generate_alias_candidates(
            {"surface": "brca1", "type": "Gene"},
            canonicalization.norm_surface(TEXT))
        self.assertEqual(_canon([list(c) for c in cands]),
                         _canon(GOLDEN["candidates_brca1"]))

    def test_qualifiers_are_pack_data(self):
        data = json.loads(
            (ROOT / "packs" / "bio-medical"
             / "fitted_qualifiers.json").read_text())
        self.assertEqual(canonicalization.QUALIFIERS,
                         frozenset(data["qualifiers"]))
        self.assertEqual(sorted(canonicalization.QUALIFIERS),
                         GOLDEN["qualifiers"])

    def test_injected_qualifiers_gate_a4(self):
        cands = canonicalization.generate_alias_candidates(
            {"surface": "brca1", "type": "Gene"},
            canonicalization.norm_surface(TEXT),
            qualifiers=frozenset())
        self.assertFalse(any(r == "A4" for _, r in cands))

    def test_placement(self):
        # The production package no longer carries the
        # instrument; the harness does.
        self.assertIsNone(
            importlib.util.find_spec("pipeline.canonicalization"))
        self.assertIsNotNone(
            importlib.util.find_spec("harness.canonicalization"))

    def test_fitted_qualifiers_candidates_cover_installed_layout(self):
        # Regression: the qualifier set is shipped
        # pack data; loading it through a repo-root-relative
        # path alone made the module unimportable from an
        # installed copy, where the packs live under the
        # install's share/ontosmith tree. The candidate list
        # must include that layout, and the in-tree candidate
        # must resolve here.
        cands = canonicalization._fitted_qualifiers_candidates()
        self.assertTrue(any(
            c.parts[-5:-2] == ("share", "ontosmith", "packs")
            for c in cands))
        self.assertTrue(any(c.is_file() for c in cands))
        self.assertEqual(canonicalization._load_fitted_qualifiers(),
                         canonicalization.QUALIFIERS)


if __name__ == "__main__":
    unittest.main()
