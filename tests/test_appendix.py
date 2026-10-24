"""Appendix integrity tests.

The staged appendix is the flagship record's evidence:
adjudication samples, keys, per-file verdicts, the sample
manifest, and the raw scores. These tests pin the shipped
files (sha256 — the same values recorded against the
lab sources at copy time), their sizes, and the
coverage contract: every sampled item carries exactly one
final judgment, and verdicts stay inside the adjudication
vocabulary.
"""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APPENDIX = ROOT / "benchmarks" / "appendix"

# sha256 of the staged files, verified identical to the
# lab sources at copy time.
EXPECTED_SHA256 = {
    "sample_a_items.jsonl":
        "353fa23fed1bebcc121baee67c4fff05a5b4e2aea9ee46fa15ef8d60e2e85269",
    "sample_a_key.jsonl":
        "003b8a0172b12c09b3079ee579e805f9102c6d77007f5173bc9135807696d634",
    "sample_b_items.jsonl":
        "2677bf2df619da6cb5e397f24ff60dc21da46e810426ea30594f1b3b65878338",
    "sample_b_key.jsonl":
        "83076cc909cec39b6a535eca94c28f1109cfac96162c9aad52de9d80c920a794",
    "judgments/a1.jsonl":
        "432b7568c00b24f6c3142f872e088c6ee99d2d392944e642c32cd6a6040f98ca",
    "judgments/a2.jsonl":
        "99b266ba8024595d98ec5541a60a5dcea15b2aa4e44e3392ac68e02a7a2be514",
    "judgments/a3.jsonl":
        "9cb7e0c3155442712cf68b2a5e23529beda82ef17fdfddbcde7b64a1f4c200e9",
    "judgments/a4.jsonl":
        "09faaee458ea2b0b392ebb3516e8f8f68552cfec1e47f0335edd38304ee988d4",
    "judgments/b1.jsonl":
        "8415a14273cb82ea6bd7ace99f9fb26e6cd569c22967bc5d69695ee59c96bc42",
    "judgments/b2.jsonl":
        "8878ef4f7b49816db098404fc81116db616a7518ff2a5029fb57f5110708890f",
    "judgments/b3.jsonl":
        "7c74b60c3c224f5c59bcc989ef7dabebf7ba98c5dd0f3934f56812da841fafbd",
    "judgments/b4.jsonl":
        "a1d7b30444776da743c05d2599ad2d2b0b9116e77dc37c350a66a177679c43a4",
    "sample_manifest.json":
        "a0ccea3d5d6e100a8a22b9e3124e5fc50a57561e872173d1951c126f27bc0133",
    "raw_scores.json":
        "7396786e3b529b2e159b2a1fbeebcf7f9446e50773400afe0c5e85a1195b4802",
    # The ruled verdict set (added 2026-10-22):
    # re-serialized per direction from the ruled run's
    # ruled_verdicts.jsonl (appendix README table).
    "judgments_ruled/a.jsonl":
        "178379e0e5943ab409b8496aced2a33dc0a5cfd8d7a41a875305a8d3f9c66ad1",
    "judgments_ruled/b.jsonl":
        "adb2aaa7e20adc1b096241827af67c88642163803289a3b0c071549a4cce9194",
    "RULING.md":
        "acff9f4f62bc1835deec788d648a761cae95db5a1a27e1fe02252dfef431ce24",
}

SEAM_CODES = {"none", "DEM", "RELPREC", "EVAL", "CONT-a",
              "CONT-b", "CONT-c", "IDENT", "QUAL"}

VERDICTS = {"supported", "not_supported", "unclear"}


def _rows(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestAppendixFiles(unittest.TestCase):
    def test_all_files_present_with_recorded_hashes(self):
        for rel, expected in EXPECTED_SHA256.items():
            path = APPENDIX / rel
            self.assertTrue(path.is_file(), f"missing {rel}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, expected, f"hash drift in {rel}")

    def test_appendix_readme_present(self):
        self.assertTrue((APPENDIX / "README.md").is_file())


class TestAppendixCoverage(unittest.TestCase):
    def _check_direction(self, items_name, prefix, shard_sizes):
        items = _rows(APPENDIX / items_name)
        item_ids = [row["item_id"] for row in items]
        self.assertEqual(len(item_ids), len(set(item_ids)),
                         "duplicate item ids")
        judged = []
        for idx, size in enumerate(shard_sizes, start=1):
            rows = _rows(APPENDIX / "judgments" / f"{prefix}{idx}.jsonl")
            self.assertEqual(len(rows), size)
            judged.extend(rows)
        judged_ids = [row["id"] for row in judged]
        self.assertEqual(sorted(judged_ids), sorted(item_ids),
                         "every sampled item must carry one judgment")
        for row in judged:
            self.assertIn(row["verdict"], VERDICTS)

    def test_direction_a_coverage(self):
        # 300 sampled + 40 controls, four queue shards of 85.
        self._check_direction("sample_a_items.jsonl", "a", [85, 85, 85, 85])

    def test_direction_b_coverage(self):
        # 400 sampled, four queue shards of 100.
        self._check_direction("sample_b_items.jsonl", "b",
                              [100, 100, 100, 100])

    def test_keys_align_with_items(self):
        for items_name, key_name in (
                ("sample_a_items.jsonl", "sample_a_key.jsonl"),
                ("sample_b_items.jsonl", "sample_b_key.jsonl")):
            items = _rows(APPENDIX / items_name)
            keys = _rows(APPENDIX / key_name)
            self.assertEqual(len(items), len(keys))
            self.assertEqual(
                sorted(r["item_id"] for r in items),
                sorted(r["item_id"] for r in keys))

    def test_ruled_verdicts_cover_the_flagged_set(self):
        # The ruled set is exactly the 565 seam-flagged items
        #: every ruled id is a sampled item, no
        # duplicates, verdicts inside the vocabulary, and
        # every row names the ruling that decided it.
        item_ids = set()
        for name in ("sample_a_items.jsonl", "sample_b_items.jsonl"):
            item_ids.update(r["item_id"] for r in _rows(APPENDIX / name))
        ruled = []
        for name in ("a.jsonl", "b.jsonl"):
            ruled.extend(_rows(APPENDIX / "judgments_ruled" / name))
        self.assertEqual(len(ruled), 565)
        ruled_ids = [r["id"] for r in ruled]
        self.assertEqual(len(ruled_ids), len(set(ruled_ids)))
        self.assertTrue(set(ruled_ids) <= item_ids)
        for row in ruled:
            self.assertIn(row["verdict"], VERDICTS)
            self.assertIn(row["seam_applied"], SEAM_CODES)

    def test_manifest_frame_matches_record(self):
        manifest = json.loads(
            (APPENDIX / "sample_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sample_a"]["n"], 300)
        self.assertEqual(manifest["sample_a"]["n_control"], 40)
        self.assertEqual(manifest["sample_b"]["n"], 400)
        self.assertEqual(
            manifest["sample_b"]["frame"]["gold_total"], 17284)
        self.assertEqual(manifest["api_spend_usd"], 0)


if __name__ == "__main__":
    unittest.main()
