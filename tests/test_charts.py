"""Chart/record agreement tests.

The figures are rendered by parsing the shipped record —
never by retyping numbers. These tests pin the parse
against the frozen record values (a drift in either the
record or the parser fails here), smoke-render every
figure, and pin the placement decision (2026-10-22):
the README carries process diagrams only; the quality
chart is a whitepaper asset under benchmarks/assets/.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "redocred.md"

# Rendering needs matplotlib (the [viz] extra); parsing the
# record does not. On a base install the parse tests still
# run and only the render smoke skips.
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None

# Support rates are the record's current figures: the
# ruled adjudication contract's rates (2026-10-22;
# whitepaper §5.3). Counts are the frozen census fates.
FROZEN_FATES = {
    "matched": (1838, 0.884),
    "never_enumerated": (6340, 0.424),
    "roster_missed": (3670, 0.529),
    "none_judged": (3677, 0.518),
    "emission_blocked": (1759, 0.500),
}


class TestAnatomyParse(unittest.TestCase):
    def test_parse_matches_frozen_record(self):
        from harness.render_charts import parse_anatomy

        parsed = parse_anatomy(RECORD.read_text(encoding="utf-8"))
        self.assertEqual(parsed["total"], 17284)
        self.assertEqual(set(parsed["fates"]), set(FROZEN_FATES))
        for fate, (count, rate) in FROZEN_FATES.items():
            self.assertEqual(parsed["fates"][fate]["count"], count, fate)
            self.assertAlmostEqual(
                parsed["fates"][fate]["rate"], rate, places=3, msg=fate)
        self.assertEqual(
            sum(f["count"] for f in parsed["fates"].values()), 17284)


@unittest.skipUnless(HAS_MATPLOTLIB,
                     "matplotlib (the [viz] extra) is not installed")
class TestRenderSmoke(unittest.TestCase):
    def _assert_png(self, path):
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 10_000)
        self.assertEqual(path.read_bytes()[:4], b"\x89PNG")

    def test_all_figures_render(self):
        import matplotlib

        matplotlib.use("Agg")
        from harness import render_charts

        anatomy = render_charts.parse_anatomy(
            RECORD.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self._assert_png(render_charts.render_anatomy(
                anatomy, out / "anatomy.png"))
            self._assert_png(render_charts.render_method(
                out / "method.png"))
            self._assert_png(render_charts.render_pipeline_flow(
                out / "pipeline-flow.png"))
            self._assert_png(render_charts.render_ontology_loop(
                out / "ontology-loop.png"))


class TestReadmePlacement(unittest.TestCase):
    def test_process_diagrams_wired_scores_not_embedded(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("docs/assets/pipeline-flow.png", readme)
        self.assertIn("docs/assets/ontology-loop.png", readme)
        self.assertIn("docs/assets/method.png", readme)
        # Placement: no score chart embedded in the README;
        # the anatomy chart is a benchmarks-side asset.
        self.assertNotIn("docs/assets/anatomy.png", readme)
        self.assertTrue(
            (ROOT / "benchmarks" / "assets" / "anatomy.png").is_file())
        for name in ("pipeline-flow.png", "ontology-loop.png",
                     "method.png"):
            self.assertTrue((ROOT / "docs" / "assets" / name).is_file())


if __name__ == "__main__":
    unittest.main()
