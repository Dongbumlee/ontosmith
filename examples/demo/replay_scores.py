"""Replay the flagship corrected scoring from shipped artifacts ($0).

Recomputes the Re-DocRED quality record's headline estimates
(canonical text: the whitepaper, §5) from the supplementary appendix
(benchmarks/appendix/): the judgment samples, their key files,
and the verdicts — using the shipped verification cores
(harness/verification/). No model calls, no network: this is
the reproduction half of the demo (docs/hands-on.md, step 5).

Two verdict sets ship in the appendix, and this replay
computes both through the identical estimator: the original
adjudication protocol's verdicts (judgments/) and the ruled
verdicts (judgments_ruled/) — the same 565 seam-flagged items
re-judged blind under the adjudication contract completed by
the six seam rulings of 2026-10-22 (docs/ADJUDICATION.md;
appendix RULING.md). The record's headlines are the ruled
estimates; the original-protocol estimates are computed
beside them as the measurement's provenance.

Method (record §Method in brief, mirrored exactly):
  * a "supported" verdict counts only with a verbatim quote
    from the item's document context (quote verification);
    quote-failed supported verdicts and unclear verdicts are
    non-decisive and excluded from every denominator;
  * Direction A: stratified estimate of the support rate over
    the unmatched Tier-1 pool (strata pair_matched /
    no_gold_pair, frame populations from the sample manifest);
    corrected precision = (exact matches + p_A x pool) /
    Tier-1 distinct, with the interval carried through p_A;
  * Direction B: stratified estimate of gold's support rate
    (strata = corpus populations); ceiling-adjusted recall =
    matched gold / (gold total x g_B);
  * CONF-440 uses the design's ratio estimator for Direction A
    (design weights = frame N / full-sample allocation) and
    the population-subset stratified estimate for Direction B;
  * the 40 blind controls are read beside the estimate, never
    inside it.

Exit code 0 iff every reproduced headline matches the
published value at 4 decimal places (intervals within 0.002).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from harness.verification.audit import quote_verified  # noqa: E402
from harness.verification.scoring import (  # noqa: E402
    corrected_precision,
    stratified_estimate,
    wilson,
    wilson_from_var,
)

APPENDIX = ROOT / "benchmarks" / "appendix"

# The published headlines (whitepaper §5) this replay
# must land on: point estimates exact at 4dp, intervals ±0.002.
PUBLISHED = {
    "p_a": (0.945, [0.916, 0.964]),
    "g_b": (0.841, [0.801, 0.874]),
    "controls": (40, 40),
    "quote_failures": 32,
    "corrected_precision_full": (0.9765, [0.9641, 0.9847]),
    "corrected_precision_conf": (0.9753, [0.9603, 0.9849]),
    "adjusted_recall_full": (0.1265, [0.1216, 0.1328]),
    "adjusted_recall_conf": (0.1263, [0.1212, 0.1332]),
    "fates": {  # fate: (sampled, support rate) — record §Anatomy
        "matched": (44, 0.977), "none_judged": (86, 0.852),
        "emission_blocked": (32, 0.774), "never_enumerated": (153, 0.787),
        "roster_missed": (85, 0.876)},
}

# The current record's headlines: the same estimator over
# the ruled verdict set (565 seam items re-judged under the
# ruled contract; the remaining 175 keep their original
# verdicts). These are the figures whitepaper §5 carries.
PUBLISHED_RULED = {
    "p_a": (0.7919, [0.747, 0.831]),
    "g_b": (0.5244, [0.475, 0.573]),
    "controls": (37, 40),
    "quote_failures": 11,
    "corrected_precision_full": (0.9110, [0.8919, 0.9275]),
    "corrected_precision_conf": (0.9121, [0.8897, 0.9310]),
    "adjusted_recall_full": (0.2028, [0.1856, 0.2237]),
    "adjusted_recall_conf": (0.2036, [0.1852, 0.2262]),
    "fates": {  # fate: (sampled, support rate) — record §Anatomy
        "matched": (44, 0.884), "none_judged": (86, 0.518),
        "emission_blocked": (32, 0.500), "never_enumerated": (153, 0.424),
        "roster_missed": (85, 0.529)},
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _derive_labels(items: dict, verdicts: dict) -> tuple[dict, int]:
    """Final decisive labels + quote-failure count for a verdict set.

    A "supported" verdict counts only with a verbatim quote
    from the item's document context; quote-failed supported
    verdicts and unclear verdicts are "unlabeled"
    (non-decisive, excluded from every denominator).
    """
    final, quote_failures = {}, 0
    for item_id, verdict in verdicts.items():
        v = verdict["verdict"]
        if v == "supported" and quote_verified(
                verdict.get("evidence", ""), items[item_id]["context"]):
            final[item_id] = "supported"
        elif v == "not_supported":
            final[item_id] = "not_supported"
        else:  # quote-failed supported, or unclear: non-decisive
            final[item_id] = "unlabeled"
            if v == "supported":
                quote_failures += 1
    return final, quote_failures


def load_appendix(appendix: Path = APPENDIX) -> dict:
    """Load samples, keys, verdicts; apply quote verification.

    Returns final decisive labels per item: "supported" /
    "not_supported" / "unlabeled" (quote-failed supported and
    unclear verdicts are both non-decisive, per the record).
    """
    items, keys, verdicts = {}, {}, {}
    for s in ("a", "b"):
        for row in _load_jsonl(appendix / f"sample_{s}_items.jsonl"):
            items[row["item_id"]] = row
        for row in _load_jsonl(appendix / f"sample_{s}_key.jsonl"):
            keys[row["item_id"]] = row
    for shard in sorted((appendix / "judgments").glob("*.jsonl")):
        for row in _load_jsonl(shard):
            verdicts[row["id"]] = row
    final, quote_failures = _derive_labels(items, verdicts)
    return {"items": items, "keys": keys, "verdicts": verdicts,
            "final": final, "quote_failures": quote_failures,
            "manifest": json.loads(
                (appendix / "sample_manifest.json").read_text(encoding="utf-8")),
            "raw": json.loads(
                (appendix / "raw_scores.json").read_text(encoding="utf-8"))}


def load_ruled_appendix(appendix: Path = APPENDIX) -> dict:
    """The appendix with the ruled verdict set substituted in.

    The ruled set (judgments_ruled/, appendix RULING.md)
    covers the 565 seam-flagged items, re-judged blind under
    the ruled contract; every other item keeps its original
    verdict. Final labels are derived through the identical
    quote-verification path as the original set.
    """
    data = load_appendix(appendix)
    verdicts = dict(data["verdicts"])
    for shard in sorted((appendix / "judgments_ruled").glob("*.jsonl")):
        for row in _load_jsonl(shard):
            verdicts[row["id"]] = {"verdict": row["verdict"],
                                   "evidence": row.get("evidence", ""),
                                   "note": row.get("note", "")}
    final, quote_failures = _derive_labels(data["items"], verdicts)
    return {**data, "verdicts": verdicts, "final": final,
            "quote_failures": quote_failures}


def _groups(keys: dict, final: dict, field: str, populations: dict,
            ids: list[str]) -> list[dict]:
    """Stratified-estimate groups for the shipped core."""
    groups = []
    for name, population in populations.items():
        sel = [i for i in ids if keys[i][field] == name]
        groups.append({"name": name, "population": population,
                       "sampled": len(sel),
                       "labels": [final[i] for i in sel]})
    return groups


def reproduce(data: dict) -> dict:
    """Recompute every published headline from the appendix."""
    keys, final = data["keys"], data["final"]
    manifest, raw = data["manifest"], data["raw"]
    a_ids = [i for i in keys if i.startswith("A-")]
    b_ids = [i for i in keys if i.startswith("B-")]
    pool_ids = [i for i in a_ids if not keys[i]["control"]]
    ctrl_ids = [i for i in a_ids if keys[i]["control"]]
    frame_a = manifest["sample_a"]["frame"]
    frame_b = manifest["sample_b"]["frame"]["by_population"]

    # Direction A, FULL-500: plain stratified estimate. The
    # core reports its estimate rounded to 4dp; the record's
    # downstream arithmetic used the unrounded value, which
    # the estimated supported count recovers to ~1e-6.
    est_a = stratified_estimate(_groups(
        keys, final, "stratum", frame_a["strata"], pool_ids))
    p_a, ci_a = est_a["estimate"], est_a["wilson95"]
    p_a_full = est_a["estimated_supported_count"] / est_a["population"]
    full = raw["FULL-500"]
    corr_full = corrected_precision(
        full["matched_extracted"], full["tier1_distinct"],
        p_a_full * frame_a["unmatched_pool"])
    corr_full_ci = [
        round((full["matched_extracted"] + ci_a[0] * frame_a["unmatched_pool"])
              / full["tier1_distinct"], 4),
        round((full["matched_extracted"] + ci_a[1] * frame_a["unmatched_pool"])
              / full["tier1_distinct"], 4)]

    # Direction A, CONF-440: the design's ratio estimator
    # (weights = frame N / full-sample allocation per stratum).
    alloc_a = manifest["sample_a"]["allocation"]
    conf_ids = [i for i in pool_ids
                if keys[i]["population"] in ("audit30", "rest410")]
    num = den = n_pool_conf = var_conf = 0.0
    for h, n_frame in frame_a["strata"].items():
        w = n_frame / alloc_a[h]
        sel = [i for i in conf_ids if keys[i]["stratum"] == h]
        lab = Counter(final[i] for i in sel)
        dec = lab["supported"] + lab["not_supported"]
        p_h = lab["supported"] / dec if dec else 0.0
        num += w * lab["supported"]
        den += w * dec
        n_pool_conf += w * len(sel)
        if dec > 1:
            share = (w * len(sel))
            var_conf += share * share * p_h * (1 - p_h) / dec
    p_a_conf = num / den if den else 0.0
    var_conf = var_conf / (n_pool_conf ** 2) if n_pool_conf else 0.0
    ci_a_conf = wilson_from_var(p_a_conf, var_conf)
    conf = raw["CONF-440"]
    pool_conf = conf["tier1_distinct"] - conf["matched_extracted"]
    corr_conf = corrected_precision(
        conf["matched_extracted"], conf["tier1_distinct"],
        p_a_conf * pool_conf)
    corr_conf_ci = [
        round((conf["matched_extracted"] + ci_a_conf[0] * pool_conf)
              / conf["tier1_distinct"], 4),
        round((conf["matched_extracted"] + ci_a_conf[1] * pool_conf)
              / conf["tier1_distinct"], 4)]

    # Direction B: stratified over corpus populations.
    est_b = stratified_estimate(_groups(
        keys, final, "population", frame_b, b_ids))
    g_b, ci_b = est_b["estimate"], est_b["wilson95"]
    g_b_full = est_b["estimated_supported_count"] / est_b["population"]
    adj_full = round(full["matched_gold"] / (full["gold_total"] * g_b_full), 4)
    adj_full_ci = [round(full["matched_gold"] / (full["gold_total"] * ci_b[1]), 4),
                   round(full["matched_gold"] / (full["gold_total"] * ci_b[0]), 4)]
    frame_b_conf = {p: frame_b[p] for p in ("audit30", "rest410")}
    est_b_conf = stratified_estimate(_groups(
        keys, final, "population", frame_b_conf,
        [i for i in b_ids if keys[i]["population"] in frame_b_conf]))
    ci_b_conf = est_b_conf["wilson95"]
    g_b_conf_full = (est_b_conf["estimated_supported_count"]
                     / est_b_conf["population"])
    adj_conf = round(conf["matched_gold"] / (conf["gold_total"] * g_b_conf_full), 4)
    adj_conf_ci = [round(conf["matched_gold"] / (conf["gold_total"] * ci_b_conf[1]), 4),
                   round(conf["matched_gold"] / (conf["gold_total"] * ci_b_conf[0]), 4)]

    # Controls (beside the estimate, never inside it).
    ctrl = Counter(final[i] for i in ctrl_ids)
    ctrl_dec = ctrl["supported"] + ctrl["not_supported"]

    # Fate support rates from Direction B (descriptive).
    fates = {}
    for fate in PUBLISHED["fates"]:
        sel = [i for i in b_ids if keys[i]["run_fate"] == fate]
        lab = Counter(final[i] for i in sel)
        dec = lab["supported"] + lab["not_supported"]
        lo, hi = wilson(lab["supported"], dec)
        fates[fate] = {"sampled": len(sel), "decisive": dec,
                       "supported": lab["supported"],
                       "rate": round(lab["supported"] / dec, 4) if dec else None,
                       "wilson95": [lo, hi]}

    return {"p_a": p_a, "ci_a": ci_a, "g_b": g_b, "ci_b": ci_b,
            "controls_supported": ctrl["supported"], "controls_decisive": ctrl_dec,
            "quote_failures": data["quote_failures"],
            "corrected_precision_full": corr_full, "corr_full_ci": corr_full_ci,
            "corrected_precision_conf": corr_conf, "corr_conf_ci": corr_conf_ci,
            "adjusted_recall_full": adj_full, "adj_full_ci": adj_full_ci,
            "adjusted_recall_conf": adj_conf, "adj_conf_ci": adj_conf_ci,
            "fates": fates}


def check(rep: dict, published: dict = PUBLISHED) -> list[str]:
    """Compare a reproduction against a published headline set."""
    failures = []

    def point(name, got, want):
        if got != want:
            failures.append(f"{name}: reproduced {got} != published {want}")

    def interval(name, got, want):
        if any(abs(g - w) > 0.002 for g, w in zip(got, want, strict=True)):
            failures.append(f"{name}: reproduced {got} vs published {want}")

    point("p_A", rep["p_a"], published["p_a"][0])
    interval("p_A CI", rep["ci_a"], published["p_a"][1])
    point("g_B", rep["g_b"], published["g_b"][0])
    interval("g_B CI", rep["ci_b"], published["g_b"][1])
    if (rep["controls_supported"], rep["controls_decisive"]) != published["controls"]:
        failures.append(f"controls: {rep['controls_supported']}/"
                        f"{rep['controls_decisive']} != "
                        f"{published['controls'][0]}/{published['controls'][1]}")
    point("quote failures", rep["quote_failures"], published["quote_failures"])
    point("corrected precision FULL-500", rep["corrected_precision_full"],
          published["corrected_precision_full"][0])
    interval("corrected precision FULL-500 CI", rep["corr_full_ci"],
             published["corrected_precision_full"][1])
    point("corrected precision CONF-440", rep["corrected_precision_conf"],
          published["corrected_precision_conf"][0])
    interval("corrected precision CONF-440 CI", rep["corr_conf_ci"],
             published["corrected_precision_conf"][1])
    point("adjusted recall FULL-500", rep["adjusted_recall_full"],
          published["adjusted_recall_full"][0])
    interval("adjusted recall FULL-500 CI", rep["adj_full_ci"],
             published["adjusted_recall_full"][1])
    point("adjusted recall CONF-440", rep["adjusted_recall_conf"],
          published["adjusted_recall_conf"][0])
    interval("adjusted recall CONF-440 CI", rep["adj_conf_ci"],
             published["adjusted_recall_conf"][1])
    for fate, (n, rate) in published["fates"].items():
        # The record publishes fate support rates at 3dp.
        got = rep["fates"][fate]
        if got["sampled"] != n or round(got["rate"], 3) != rate:
            failures.append(f"fate {fate}: reproduced {got['sampled']}/"
                            f"{got['rate']} != published {n}/{rate}")
    return failures


def _print_card(title: str, rep: dict) -> None:
    print(title)
    print(f"  Direction A support estimate  p_A = {rep['p_a']} {rep['ci_a']}")
    print(f"  Direction B gold support       g_B = {rep['g_b']} {rep['ci_b']}")
    print(f"  Blind controls                 {rep['controls_supported']}/"
          f"{rep['controls_decisive']} supported")
    print(f"  Quote-verification failures    {rep['quote_failures']}")
    print(f"  Corrected precision  FULL-500  {rep['corrected_precision_full']} "
          f"{rep['corr_full_ci']}")
    print(f"  Corrected precision  CONF-440  {rep['corrected_precision_conf']} "
          f"{rep['corr_conf_ci']}")
    print(f"  Ceiling-adj. recall  FULL-500  {rep['adjusted_recall_full']} "
          f"{rep['adj_full_ci']}")
    print(f"  Ceiling-adj. recall  CONF-440  {rep['adjusted_recall_conf']} "
          f"{rep['adj_conf_ci']}")


def main(argv=None) -> int:
    """Reproduce both headline sets, print both cards, exit on match."""
    rep_ruled = reproduce(load_ruled_appendix())
    rep_original = reproduce(load_appendix())
    _print_card("Corrected-scoring reproduction — from benchmarks/appendix/\n"
                "Ruled adjudication contract (the record's headlines, "
                "whitepaper §5):", rep_ruled)
    _print_card("Original adjudication protocol (provenance — the "
                "estimate as first published):", rep_original)
    failures = (check(rep_ruled, PUBLISHED_RULED)
                + check(rep_original, PUBLISHED))
    if failures:
        print("MISMATCH against the published record (whitepaper §5):")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All headlines reproduce the published record "
          "(whitepaper §5), both verdict sets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
