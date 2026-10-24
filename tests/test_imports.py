"""Import smoke: every module in the tree imports cleanly.

Baseline: 40 pipeline modules verified; later additions
brought reference_profile, gate_cli and evaluate (43) and
ontology_builder, and canonicalization moved to the
harness (a measurement instrument, not a pipeline stage),
for 43. The harness doc-check tool imports with the
standard library only.
"""
import importlib
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestPipelineImports(unittest.TestCase):
    def test_all_pipeline_modules_import(self):
        names = sorted(p.stem for p in (ROOT / "pipeline").glob("*.py"))
        self.assertGreaterEqual(len(names), 43)
        failed = {}
        for name in names:
            try:
                importlib.import_module(f"pipeline.{name}")
            except Exception as e:
                failed[name] = f"{type(e).__name__}: {e}"
        self.assertEqual(failed, {})


class TestNoSysPathPollution(unittest.TestCase):
    """Regression: lab modules used to sys.path.insert their own
    directory and import siblings by bare name, which polluted
    the process path and split module identity (a top-level
    `analyze_hardened` distinct from pipeline.analyze_hardened).
    Package modules import relatively and register no bare
    sibling names."""

    def test_imports_do_not_pollute(self):
        code = (
            "import sys\n"
            "before = list(sys.path)\n"
            "import pipeline.threshold_stability\n"
            "import pipeline.recalibrate\n"
            "import pipeline.evaluate\n"
            "assert sys.path == before, 'sys.path mutated'\n"
            "assert 'analyze_hardened' not in sys.modules\n"
            "assert 'pipeline.analyze_hardened' in sys.modules\n"
            "print('clean')\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT, capture_output=True, text=True,
            env={"PATH": "/usr/bin:/bin",
                 "PYTHONPATH": str(ROOT)})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "clean")


class TestHarnessDocCheck(unittest.TestCase):
    def test_doc_check_imports_stdlib_only(self):
        out = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'harness');"
             "import doc_check; print('ok')"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
