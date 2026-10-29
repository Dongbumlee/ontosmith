"""— the role-word guard vocabulary is pack data.

The word set was once a literal in core proposer code
(the CUAD-flavored set). It now ships as
the Legal Contract pack's role_words artefact, resolved by
profile_config and injected into proposer.guard_triples.

The golden (tests/golden/role_word_guard.json) was captured
from the PRE-CHANGE code (bool switch + core literal) on the
fixture below; every guard assertion here compares against
it byte-for-byte.

Regeneration record: the "off" entry was regenerated when
identity resolution landed in guard_triples (fragment
endpoints resolve to the document's canonical census
surface: this fixture's "Company" -> "The Company" and
"Affiliate" -> "The Affiliate", each recorded in the kept
record's `canonicalized`). The "on" entry is unchanged —
role-word blocks fire before resolution, so the guard-on
path never reaches it. The equivalence this file pins is
unchanged: role_words on/off differ only by the role-word
guard's own verdicts.
"""
import json
import shutil
import unittest
from pathlib import Path

from pipeline import profile_config, proposer

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = json.loads(
    (ROOT / "tests" / "golden" / "role_word_guard.json")
    .read_text())
FORMER_CORE_SET = frozenset(GOLDEN["former_core_role_words"])

DOC = (
    "The Company employs John Smith. Party A owns Globex. "
    "The Service Provider advises Acme Corp. Acme Corp acquired "
    "Globex. The Affiliate reports to John Smith. "
    "The company announced earnings."
)
ITEMS = [
    {"subject": "Company", "relation": "employs",
     "object": "John Smith", "quote": "The Company employs John Smith"},
    {"subject": "Party A", "relation": "owns",
     "object": "Globex", "quote": "Party A owns Globex"},
    {"subject": "Service Provider", "relation": "advises",
     "object": "Acme Corp",
     "quote": "The Service Provider advises Acme Corp"},
    {"subject": "Acme Corp", "relation": "acquired",
     "object": "Globex", "quote": "Acme Corp acquired Globex"},
    {"subject": "Affiliate", "relation": "reports to",
     "object": "John Smith",
     "quote": "The Affiliate reports to John Smith"},
    {"subject": "The Company", "relation": "announced",
     "object": "earnings", "quote": "The company announced earnings"},
]


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


class TestArtefact(unittest.TestCase):
    def test_vocabulary_equals_former_core_set(self):
        vocab = profile_config.pack_role_words(
            ROOT / "packs" / "legal-contract")
        self.assertEqual(vocab, FORMER_CORE_SET)

    def test_other_packs_ship_no_artefact(self):
        for slug in ("core", "open-domain", "bio-medical"):
            self.assertIsNone(profile_config.pack_role_words(
                ROOT / "packs" / slug), slug)

    def test_pin_drift_refuses(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "packs" / "legal-contract"
            dst.mkdir(parents=True)
            src = ROOT / "packs" / "legal-contract"
            shutil.copy(src / "pack.json", dst / "pack.json")
            drifted = (src / "role_words.json").read_text() \
                .replace("affiliate", "affiliatex")
            (dst / "role_words.json").write_text(drifted)
            with self.assertRaises(ValueError):
                profile_config.pack_role_words(dst)


class TestGuardEquivalence(unittest.TestCase):
    def test_guard_on_byte_identical(self):
        vocab = profile_config.pack_role_words(
            ROOT / "packs" / "legal-contract")
        kept, blocked = proposer.guard_triples(
            ITEMS, DOC, role_words=vocab)
        self.assertEqual(_canon({"kept": kept, "blocked": blocked}),
                         _canon(GOLDEN["on"]))

    def test_guard_off_byte_identical(self):
        kept, blocked = proposer.guard_triples(
            ITEMS, DOC, role_words=None)
        self.assertEqual(_canon({"kept": kept, "blocked": blocked}),
                         _canon(GOLDEN["off"]))

    def test_core_carries_no_vocabulary(self):
        self.assertFalse(hasattr(proposer, "ROLE_WORDS"))
        src = (ROOT / "pipeline" / "proposer.py").read_text()
        self.assertNotIn("ROLE_WORDS", src)
        self.assertNotIn('"service provider"', src)


class TestResolution(unittest.TestCase):
    def test_resolution_freezes_vocabulary(self):
        res = profile_config.resolve_setup_config(
            ROOT, ROOT / "packs" / "legal-contract")
        self.assertEqual(res["role_words"], sorted(FORMER_CORE_SET))
        for slug in ("open-domain", "bio-medical"):
            res = profile_config.resolve_setup_config(
                ROOT, ROOT / "packs" / slug)
            self.assertNotIn("role_words", res, slug)

    def test_effective_role_words_paths(self):
        payload = profile_config.pack_setup_config(
            ROOT / "packs" / "legal-contract")
        # family path (unfrozen payload, no role_words key)
        self.assertEqual(
            profile_config.effective_role_words(ROOT, payload),
            FORMER_CORE_SET)
        # frozen path (resolved config carries the key)
        frozen = {**payload, "role_words": sorted(FORMER_CORE_SET)}
        self.assertEqual(
            profile_config.effective_role_words(ROOT, frozen),
            FORMER_CORE_SET)
        # non-declaring config: guard off
        other = profile_config.pack_setup_config(
            ROOT / "packs" / "bio-medical")
        self.assertIsNone(
            profile_config.effective_role_words(ROOT, other))
        self.assertIsNone(
            profile_config.effective_role_words(ROOT, None))

    def test_declared_without_vocabulary_refuses(self):
        cfg = {"role_word_rules_from_profile": ["a rule"],
               "family": "no-such-family"}
        with self.assertRaises(ValueError):
            profile_config.effective_role_words(ROOT, cfg)

    def test_registry_and_typecheck(self):
        self.assertIn("role_words",
                      profile_config.CONSUMED_KEYS)
        errs = profile_config.setup_config_errors(
            {"role_words": "not-a-list"})
        self.assertTrue(any("role_words" in e for e in errs))
        self.assertEqual(profile_config.setup_config_errors(
            {"role_words": ["company"]}), [])


if __name__ == "__main__":
    unittest.main()
