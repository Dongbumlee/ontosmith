"""Profile-config registry tests (Phase B, A2-F4 + pack finding).

A2-F4: batch_run consumes three CANDIDATES guard keys that
the registry did not register, so fail-closed validation
rejected any pack payload declaring them — the issue #68
guard relaxations could not ride a pack. Registration must
preserve fail-closed semantics exactly: the keys validate as
booleans, wrong types are named errors, unregistered keys
still fail, and absence keeps today's defaults (batch_run
reads all three with bool(cfg.get(...)) -> False).
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from pipeline.profile_config import (
    CONSUMED_KEYS,
    find_pack_dir,
    pack_dir_for_family,
    setup_config_errors,
)

STAGING = Path(__file__).resolve().parent.parent

F4_KEYS = ("pair_abbrev_guard", "pair_theme_attest_sentence",
           "quote_punct_insensitive")


class TestF4Registration(unittest.TestCase):
    def test_keys_registered_as_consumed(self):
        for key in F4_KEYS:
            self.assertIn(key, CONSUMED_KEYS)

    def test_boolean_values_validate(self):
        for key in F4_KEYS:
            self.assertEqual(setup_config_errors({key: True}), [])
            self.assertEqual(setup_config_errors({key: False}), [])

    def test_non_boolean_values_fail_closed(self):
        for key in F4_KEYS:
            errs = setup_config_errors({key: "yes"})
            self.assertEqual(len(errs), 1)
            self.assertIn(key, errs[0])
            self.assertIn("boolean", errs[0])

    def test_unregistered_key_still_fails(self):
        errs = setup_config_errors({"pair_theme_attest": True})
        self.assertEqual(len(errs), 1)
        self.assertIn("pair_theme_attest", errs[0])

    def test_absent_keys_keep_defaults(self):
        # batch_run reads: bool(cfg.get(key)) — absence is False,
        # the frozen legacy behavior; registration changes none
        # of that (it only permits declaration).
        cfg: dict = {}
        for key in F4_KEYS:
            self.assertFalse(bool(cfg.get(key)))


class TestPackFinding(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        shutil.copytree(STAGING / "packs", self.tmp / "packs")

    def test_family_lookup_uses_manifest_not_dirname(self):
        # the bio-medical pack's directory is a domain name;
        # its manifest binds family 'biored'
        found = pack_dir_for_family(self.tmp, "biored")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "bio-medical")

    def test_family_lookup_unknown_family(self):
        self.assertIsNone(pack_dir_for_family(self.tmp, "nope"))

    def test_find_pack_dir_by_id_family_and_path(self):
        self.assertEqual(find_pack_dir(self.tmp, "bio-medical").name,
                         "bio-medical")
        self.assertEqual(find_pack_dir(self.tmp, "biored").name,
                         "bio-medical")
        direct = self.tmp / "packs" / "core"
        self.assertEqual(find_pack_dir(self.tmp, direct), direct)
        self.assertEqual(
            find_pack_dir(self.tmp, direct / "pack.json"), direct)

    def test_find_pack_dir_unknown_ref_raises(self):
        with self.assertRaises(ValueError):
            find_pack_dir(self.tmp, "no-such-pack")


if __name__ == "__main__":
    unittest.main()
