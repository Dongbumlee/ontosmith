"""Pack resolution from an installed layout (C-F4).

A wheel install places the pack tree at the installed
data-files location — ``<sysconfig data>/share/ontosmith
/packs`` — not under the pipeline root (site-packages in
that layout). Run 10 recorded the defect: a pack-bound
``batch create`` from such an install failed because the
pack lookups consulted only ``<root>/packs``.

These tests simulate the installed layout with plain
directories: a root with no ``packs/`` of its own, and a
copy of the real pack tree standing at the installed
location (``installed_packs_dir`` patched to it). The
source tree keeps precedence wherever it has packs of its
own — the existing suites pin that behavior; the precedence
test here pins that the installed tree never shadows it.
"""
from __future__ import annotations

import shutil
import sysconfig
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import profile_config as pc

ROOT = Path(__file__).resolve().parent.parent


class TestInstalledPacksDir(unittest.TestCase):
    def test_default_location_is_the_data_files_path(self):
        expected = Path(sysconfig.get_path("data")) / \
            "share" / "ontosmith" / "packs"
        self.assertEqual(pc.installed_packs_dir(), expected)


class InstalledLayoutCase(unittest.TestCase):
    """A root with no packs + a simulated installed tree."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        # The pipeline root of an installed layout: code
        # only, no packs/ beside it.
        self.root = base / "site-packages"
        self.root.mkdir()
        # The installed data-files tree.
        self.installed = base / "venv" / "share" / \
            "ontosmith" / "packs"
        self.installed.parent.mkdir(parents=True)
        shutil.copytree(ROOT / "packs", self.installed,
                        ignore=shutil.ignore_patterns(
                            "__pycache__"))
        patcher = mock.patch.object(
            pc, "installed_packs_dir",
            return_value=self.installed)
        patcher.start()
        self.addCleanup(patcher.stop)


class TestFindPackDirInstalled(InstalledLayoutCase):
    def test_by_directory_name(self):
        found = pc.find_pack_dir(self.root, "open-domain")
        self.assertEqual(found,
                         self.installed / "open-domain")

    def test_by_pack_id_and_family(self):
        self.assertEqual(
            pc.find_pack_dir(self.root, "legal-contract"),
            self.installed / "legal-contract")
        self.assertEqual(
            pc.find_pack_dir(self.root, "cuad"),
            self.installed / "legal-contract")

    def test_unknown_reference_still_refuses(self):
        with self.assertRaises(ValueError):
            pc.find_pack_dir(self.root, "no-such-pack")


class TestFamilyResolutionInstalled(InstalledLayoutCase):
    def test_pack_dir_for_family(self):
        self.assertEqual(
            pc.pack_dir_for_family(self.root, "redocred"),
            self.installed / "open-domain")
        self.assertEqual(
            pc.pack_dir_for_family(self.root, "biored"),
            self.installed / "bio-medical")
        self.assertIsNone(
            pc.pack_dir_for_family(self.root, "no-family"))

    def test_profile_setup_config_path(self):
        path = pc.profile_setup_config_path(self.root, "redocred")
        self.assertEqual(
            path, self.installed / "open-domain" /
            "profile" / "setup_config.json")

    def test_pack_for_setup_config(self):
        payload = (self.installed / "open-domain" /
                   "profile" / "setup_config.json").read_bytes()
        prov = pc.pack_for_setup_config(
            self.root, "redocred", payload)
        self.assertIsNotNone(prov)
        self.assertEqual(prov["pack_id"], "open-domain")


class TestResolveSetupConfigInstalled(InstalledLayoutCase):
    def test_baseline_only_from_installed_tree(self):
        cfg = pc.resolve_setup_config(self.root)
        self.assertTrue(cfg)  # core baseline payload

    def test_overlay_merges_installed_baseline(self):
        pack_dir = pc.find_pack_dir(self.root,
                                    "open-domain")
        cfg = pc.resolve_setup_config(self.root, pack_dir)
        baseline = pc.pack_setup_config(
            self.installed / "core")
        overlay = pc.pack_setup_config(pack_dir)
        for key, value in overlay.items():
            self.assertEqual(cfg[key], value)
        for key, value in baseline.items():
            if key not in overlay:
                self.assertEqual(cfg[key], value)

    def test_role_words_pack_resolves_installed(self):
        # legal-contract declares the role-word guard and
        # ships its vocabulary as pack data (A2-F2); the
        # resolution must freeze it from the installed tree.
        pack_dir = pc.find_pack_dir(self.root, "legal-contract")
        cfg = pc.resolve_setup_config(self.root, pack_dir)
        self.assertIn("role_words", cfg)
        self.assertTrue(cfg["role_words"])


class TestSourceTreePrecedence(unittest.TestCase):
    def test_root_packs_win_over_installed(self):
        with tempfile.TemporaryDirectory() as tmp:
            installed = Path(tmp) / "share" / "ontosmith" / \
                "packs"
            installed.parent.mkdir(parents=True)
            shutil.copytree(ROOT / "packs", installed,
                            ignore=shutil.ignore_patterns(
                                "__pycache__"))
            with mock.patch.object(
                    pc, "installed_packs_dir",
                    return_value=installed):
                self.assertEqual(
                    pc.find_pack_dir(ROOT, "open-domain"),
                    ROOT / "packs" / "open-domain")
                self.assertEqual(
                    pc.pack_dir_for_family(ROOT, "redocred"),
                    ROOT / "packs" / "open-domain")


if __name__ == "__main__":
    unittest.main()
