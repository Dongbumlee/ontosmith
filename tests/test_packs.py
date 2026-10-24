"""Pack tests: public names, loader equivalence, profile lookup.

Loader equivalence is pinned against goldens captured from the
LAB pack loader (pilot/profile_config.py) before the rename:
the payload dict a staged pack loads must be byte-identical
(canonical JSON) to the lab's, and the provenance triple must
match on every field except pack_id — which now carries the
public slug per the D7 naming rule (dataset names are
provenance, not pack names).
"""
import json
import unittest
from pathlib import Path

from pipeline.profile_config import (
    pack_for_setup_config,
    pack_setup_config,
    profile_setup_config_path,
)

ROOT = Path(__file__).resolve().parent.parent
GOLD = Path(__file__).parent / "golden"

PACKS = {
    "open-domain": ("docred", "redocred", "Re-DocRED"),
    "bio-medical": ("biored", "biored", "BioRED"),
    "legal-contract": ("cuad", "cuad", "CUAD"),
}


class TestPublicLayout(unittest.TestCase):
    def test_public_dirs_exist_codenames_gone(self):
        packs = ROOT / "packs"
        for slug in ("open-domain", "bio-medical",
                     "legal-contract", "core"):
            self.assertTrue((packs / slug).is_dir(), slug)
        for codename in ("docred", "biored", "cuad"):
            self.assertFalse((packs / codename).exists(), codename)

    def test_manifest_identity_fields(self):
        for slug, (_codename, _family, dataset) in PACKS.items():
            m = json.loads(
                (ROOT / "packs" / slug / "pack.json").read_text())
            self.assertEqual(m["pack_id"], slug)
            self.assertEqual(m["validated_on"], dataset)
            # Lab-provenance fields do not ship (run 29):
            # the codename lives in the lab record and in
            # these goldens' filenames, never in the manifest.
            self.assertNotIn("lab_codename", m)
            self.assertNotIn("issue", m)
            # the payload path must resolve inside the staged tree
            self.assertTrue((ROOT / m["payload"]["path"]).exists(),
                            m["payload"]["path"])


class TestLoaderEquivalence(unittest.TestCase):
    def test_payload_dict_identical_to_lab(self):
        for slug, (codename, _family, _ds) in PACKS.items():
            cfg = pack_setup_config(ROOT / "packs" / slug)
            canon = json.dumps(cfg, sort_keys=True)
            golden = (GOLD / f"pack_{codename}_cfg.json").read_text()
            self.assertEqual(canon, golden, slug)

    def test_provenance_binds_under_public_slug(self):
        for slug, (codename, family, _ds) in PACKS.items():
            payload = (ROOT / "packs" / slug / "profile"
                       / ("setup_config_v4.json"
                          if slug == "bio-medical"
                          else "setup_config.json")).read_bytes()
            prov = pack_for_setup_config(ROOT, family, payload)
            golden = json.loads(
                (GOLD / f"pack_{codename}_prov.json").read_text())
            self.assertIsNotNone(prov, slug)
            for k, v in golden.items():
                if k == "pack_id":
                    continue
                self.assertEqual(prov[k], v, f"{slug}:{k}")
            self.assertEqual(prov["pack_id"], slug)


class TestFittedQualifiers(unittest.TestCase):
    def test_qualifiers_load_from_public_pack(self):
        # harness.canonicalization.QUALIFIERS is Bio Medical
        # pack data, loaded at import from packs/bio-medical/
        # (fail-hard). The module is a harness measurement
        # instrument since run 7 (placement decision, #73).
        from harness import canonicalization
        data = json.loads(
            (ROOT / "packs" / "bio-medical"
             / "fitted_qualifiers.json").read_text())
        self.assertEqual(canonicalization.QUALIFIERS,
                         frozenset(data["qualifiers"]))
        self.assertTrue(canonicalization.QUALIFIERS)


class TestProfileLookup(unittest.TestCase):
    def test_families_resolve_in_pack_layout(self):
        expect = {
            "redocred": "open-domain",
            "biored": "bio-medical",
            "cuad": "legal-contract",
        }
        for family, slug in expect.items():
            p = profile_setup_config_path(ROOT, family)
            self.assertIsNotNone(p, family)
            self.assertEqual(p, ROOT / "packs" / slug / "profile"
                             / "setup_config.json")

    def test_unknown_family_returns_none(self):
        self.assertIsNone(profile_setup_config_path(ROOT, "fiction"))
        self.assertIsNone(profile_setup_config_path(ROOT, "nope"))


if __name__ == "__main__":
    unittest.main()
