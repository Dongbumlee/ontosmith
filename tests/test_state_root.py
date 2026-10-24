"""Batch state root.

Batch state — the batches/ tree, the registry — belongs
to the user's project, not to the installed package tree.
The clean room proved the wheel route writing batches
into site-packages. state_root() is the seam: env
override, else the source checkout beside the code, else
the working directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pipeline import batch_cli
from pipeline import profile_config as pc

ROOT = Path(__file__).resolve().parent.parent


class TestStateRootRule(unittest.TestCase):

    def test_env_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                    os.environ, {"ONTOSMITH_ROOT": tmp}):
                self.assertEqual(batch_cli.state_root(),
                                 Path(tmp))

    def test_source_checkout_is_the_state_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "packs").mkdir()
            with mock.patch.object(batch_cli, "ROOT", root):
                with mock.patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("ONTOSMITH_ROOT", None)
                    self.assertEqual(batch_cli.state_root(), root)

    def test_installed_layout_uses_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            code = base / "site-packages"  # no packs/ beside it
            code.mkdir()
            work = base / "work"
            work.mkdir()
            old = Path.cwd()
            try:
                os.chdir(work)
                with mock.patch.dict(os.environ):
                    os.environ.pop("ONTOSMITH_ROOT", None)
                    with mock.patch.object(batch_cli, "ROOT", code):
                        self.assertEqual(batch_cli.state_root(), work)
            finally:
                os.chdir(old)


class TestCreateLandsInStateRoot(unittest.TestCase):
    """The installed layout, one step further: creation from an
    installed tree resolves packs from the installed data
    tree but writes the batch under the working directory."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        self.code = base / "site-packages"  # code only
        self.code.mkdir()
        self.installed = base / "venv" / "share" / "ontosmith" / "packs"
        self.installed.parent.mkdir(parents=True)
        shutil.copytree(ROOT / "packs", self.installed,
                        ignore=shutil.ignore_patterns("__pycache__"))
        patcher = mock.patch.object(pc, "installed_packs_dir",
                                    return_value=self.installed)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.work = base / "work"
        self.work.mkdir()
        (self.work / "doc_a.txt").write_text(
            "Acme Corp was founded by Jane Doe in 2010.")
        (self.work / "profile.json").write_text(json.dumps(
            {"family": "redocred", "profile_hash": "cleanroomhash"}))

    def test_create_writes_batch_under_cwd_not_code_tree(self):
        args = argparse.Namespace(
            source="src", collection="coll",
            docs=[str(self.work / "*.txt")],
            profile=str(self.work / "profile.json"),
            batch_id="cr-batch")
        old = Path.cwd()
        try:
            os.chdir(self.work)
            with mock.patch.dict(os.environ):
                os.environ.pop("ONTOSMITH_ROOT", None)
                with mock.patch.object(batch_cli, "ROOT", self.code):
                    batch_cli.cmd_create(args)
        finally:
            os.chdir(old)
        batch_json = self.work / "batches" / "cr-batch" / "batch.json"
        self.assertTrue(batch_json.exists())
        self.assertTrue(
            (self.work / "batches" / "cr-batch"
             / "setup_config.resolved.json").exists())
        self.assertFalse((self.code / "batches").exists())
        record = json.loads(batch_json.read_text())
        self.assertEqual(record["pack_provenance"]["pack_id"],
                         "open-domain")


if __name__ == "__main__":
    unittest.main()
