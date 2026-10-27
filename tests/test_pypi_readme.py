"""PyPI README variant (scripts/build_pypi_readme.py).

PyPI does not resolve repository-relative links or
images, so the GitHub master README cannot ship as the
long description. The generator produces the variant;
this suite pins its contract: the rewrite counts equal
the audited inventory of the master, every rewritten
target is absolute under the declared base, the
install wording is the PyPI form, the brand banner
sits under the title as an absolute raw URL, hostile
masters are refused (never silently shipped), and the
committed pyproject keeps pointing at the master —
the publish-time selection swaps file content in the
release checkout, never the metadata.

One exception: the README badge row (PyPI version,
PyPI Python versions, CI status) is external by design
— shields.io renders the PyPI badges and GitHub serves
the CI badge — so those image URLs are the master's
first targets outside the repo base. They pass through
the variant untouched (PyPI renders shields badges
natively); the two tests that pin "every target under
the base" allowlist exactly the badge image URLs
below, nothing else.
"""
from __future__ import annotations

import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_pypi_readme", ROOT / "scripts" / "build_pypi_readme.py")
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

_LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def _master() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def _targets(text: str) -> list[str]:
    return _LINK_RE.findall(text)


# The run-30 badge row's image URLs: none of them is a
# rewritten target (see module docstring). The shields.io
# pair is external; the CI badge image sits under the
# real repo base but is pinned there by design — a CI
# badge must point at the real repo whatever base the
# variant is generated with.
BADGE_IMAGE_URLS = {
    "https://img.shields.io/pypi/v/ontosmith",
    "https://img.shields.io/pypi/pyversions/ontosmith",
    "https://github.com/Dongbumlee/ontosmith"
    "/actions/workflows/ci.yml/badge.svg",
}

# The start table's deliberate external destinations:
# the three films on YouTube and the project site's
# Watch section. Like the badges, they are absolute by
# design and pass through the variant unchanged —
# wherever the reader is (GitHub or PyPI), the films
# live at these URLs, not under the repo base.
EXTERNAL_LINK_URLS = {
    "https://www.youtube.com/watch?v=qmCFIIOCCqA",
    "https://www.youtube.com/watch?v=Vz2SIyquMXQ",
    "https://www.youtube.com/watch?v=PO6F9QdRxR8",
    "https://dongbumlee.github.io/ontosmith/#watch",
}


def _relative(text: str) -> list[str]:
    return [t for t in _targets(text)
            if not t.startswith(("http://", "https://", "mailto:"))]


class TestVariant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = _master()
        cls.variant, cls.counts = gen.build_variant(cls.master)

    def test_master_audits_clean(self):
        self.assertEqual(gen.audit(self.master), [])

    def test_rewrite_counts_equal_the_audit(self):
        # 19 relative links (18 files + the benchmarks/
        # directory) and 4 relative images in the master
        # — the audit counted 21 targets (3
        # images); the brand banner is the
        # fourth image; a later revision added the anchored
        # Optimized-for-Jev link to Connections (a file
        # target, rewritten to blob form with its
        # fragment preserved). Four more file
        # targets followed: the PACK_DESIGN contract link, the
        # Connections link in the reference-profile
        # section, and two hands-on links in the start
        # table. The built-graph figure followed — the fifth
        # relative image. Two more targets: the llms.txt link
        # in the start table
        # (a file target) and the reproduce/ directory
        # link in step 2 (the second tree target). Two
        # more file targets: the
        # Connections links carrying the Neo4j
        # prerequisite, in the Requirements list and in
        # step 3. The build-loop animation under the
        # intro followed — the sixth relative image
        # (an animated WebP, rewritten to the raw form
        # like every other figure). Two more file
        # targets: the hands-on links in the start
        # table's ontology-import row and in the FIBO
        # Contracts row of the corpus-packs table
        # (Track 4).
        self.assertEqual(self.counts,
                         {"blob": 28, "tree": 2, "raw": 6})
        self.assertEqual(len(_relative(self.master)), 36)

    def test_no_relative_target_survives(self):
        self.assertEqual(_relative(self.variant), [])

    def test_rewritten_targets_are_absolute_under_the_base(self):
        for target in _targets(self.variant):
            if target in BADGE_IMAGE_URLS | EXTERNAL_LINK_URLS:
                continue  # badges + film/site links: external by design
            self.assertTrue(
                target.startswith(gen.DEFAULT_BASE + "/"),
                target)

    def test_base_is_a_single_parameter(self):
        variant, _ = gen.build_variant(
            self.master, base="https://example.invalid/o/r")
        self.assertTrue(all(
            t.startswith("https://example.invalid/o/r/")
            for t in _targets(variant)
            if t not in BADGE_IMAGE_URLS | EXTERNAL_LINK_URLS))

    def test_images_use_the_raw_form(self):
        for name in ("pipeline-flow.png", "method.png",
                     "ontology-loop.png", "built-graph.png",
                     "build-loop.webp"):
            self.assertIn(
                f"{gen.DEFAULT_BASE}/raw/main/docs/assets/{name}",
                self.variant)
        self.assertIn(
            f"{gen.DEFAULT_BASE}/raw/main/{gen.BANNER_IMAGE}",
            self.variant)

    def test_directory_link_uses_the_tree_form(self):
        self.assertIn(
            f"{gen.DEFAULT_BASE}/tree/main/benchmarks",
            self.variant)

    def test_banner_sits_under_the_title(self):
        self.assertTrue(self.variant.startswith("# OntoSmith\n"))
        banner = (f"![OntoSmith]({gen.DEFAULT_BASE}"
                  f"/raw/main/{gen.BANNER_IMAGE})")
        self.assertIn(banner, self.variant)
        self.assertLess(self.variant.index(banner),
                        self.variant.index("**An ontology-driven"))
        self.assertNotIn("PYPI-BANNER-SLOT", self.variant)

    def test_install_wording_is_the_pypi_form(self):
        self.assertIn("$ pip install ontosmith", self.variant)
        self.assertIn("Install from PyPI:", self.variant)
        self.assertNotIn("pip install .", self.variant)
        self.assertNotIn("at launch this", self.variant)
        self.assertNotIn("clone of this repository",
                         self.variant)

    def test_variant_is_not_a_valid_input(self):
        # Feeding the variant back in must refuse: its
        # install wording is already the PyPI form and
        # its banner already absolute, so the anchored
        # checks no longer hold.
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant(self.variant)


class TestRefusals(unittest.TestCase):
    def test_raw_html_in_prose_refused(self):
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant("# T\n\n<div>hi</div>\n")

    def test_github_alert_refused(self):
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant("# T\n\n> [!NOTE]\n> hi\n")

    def test_fragment_only_link_refused(self):
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant("# T\n\n[up](#t)\n")

    def test_missing_banner_refused(self):
        master = _master().replace(gen.BANNER_LINE, "")
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant(master)

    def test_missing_substitution_anchor_refused(self):
        master = _master().replace(
            "Install from PyPI:",
            "Install from a mirror:")
        with self.assertRaises(gen.AuditFailure):
            gen.build_variant(master)

    def test_pyproject_must_point_at_the_master(self):
        committed = (ROOT / "pyproject.toml").read_text(
            encoding="utf-8")
        gen.check_pyproject_master(committed)  # no raise
        with self.assertRaises(gen.AuditFailure):
            gen.check_pyproject_master(
                committed.replace('readme = "README.md"',
                                  'readme = "OTHER.md"'))

    def test_main_refuses_to_overwrite_the_master(self):
        master = ROOT / "README.md"
        rc = gen.main(["--readme", str(master),
                       "--out", str(master)])
        self.assertEqual(rc, 2)

    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pypi" / "README.md"
            rc = gen.main(["--out", str(out)])
            self.assertEqual(rc, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn(
                f"![OntoSmith]({gen.DEFAULT_BASE}"
                f"/raw/main/{gen.BANNER_IMAGE})", text)
            self.assertEqual(_relative(text), [])


if __name__ == "__main__":
    unittest.main()
