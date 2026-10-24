"""The one-command evaluation report (plan §7 R10).

The lab produced its evaluation numbers per run, by hand, in
run-specific scripts; the method was proven but absent as a
product. This module is the product form: one builder that
takes predicted + gold records (and optionally the audit
verdicts) and returns the full evaluation record — raw
scores, audit-corrected precision, ceiling-adjusted recall,
and the failure anatomy — as JSON-able data plus a Markdown
rendering in the report-card shape (plan §7 R15).

Failure anatomy: every gold record is assigned exactly one
fate so the buckets sum to the gold total — ``matched``,
or a caller-supplied fate for the unmatched (e.g.
``never_enumerated`` / ``judged_none`` / ``emission_blocked``
from the caller's pipeline stages), defaulting to
``unmatched``. The anatomy is where a report earns its keep:
it says WHERE the losses are, in counts, so the next
investment is a measurement, not an opinion.
"""
from __future__ import annotations

from .scoring import (
    NormalizationTable,
    ceiling_adjusted_recall,
    corrected_precision,
    match_records,
    stratified_estimate,
)


def build_report(predicted: list[dict], gold: list[dict],
                 table: NormalizationTable | None = None,
                 unmatched_support: dict | None = None,
                 gold_support: dict | None = None,
                 gold_fates: dict | None = None) -> dict:
    """Build the evaluation record.

    ``unmatched_support`` / ``gold_support`` are
    :func:`~harness.verification.scoring.stratified_estimate`
    inputs (strata groups) OR ready estimate dicts; when
    given as groups they are estimated here and the estimate
    feeds the corrected figures. ``gold_fates`` maps a gold
    record's index (as a string) to its unmatched fate.
    """
    m = match_records(predicted, gold, table)
    report: dict = {"raw": m.raw_scores()}

    def _estimate(spec):
        if spec is None:
            return None
        if isinstance(spec, dict) and "groups" in spec \
                and "estimate" not in spec:
            return stratified_estimate(spec["groups"])
        return spec

    us = _estimate(unmatched_support)
    gs = _estimate(gold_support)
    corrected: dict = {}
    if us is not None:
        supported_count = us["estimated_supported_count"]
        corrected["precision"] = corrected_precision(
            m.matched_predicted, m.predicted_total, supported_count)
        corrected["unmatched_support"] = us
    if gs is not None:
        corrected["ceiling_adjusted_recall"] = ceiling_adjusted_recall(
            m.matched_gold, m.gold_total, gs["estimate"])
        corrected["gold_support"] = gs
    if corrected:
        report["corrected"] = corrected

    fates = gold_fates or {}
    unmatched_idx = set(m.unmatched_gold_indices)
    anatomy: dict[str, int] = {}
    for i in range(len(gold)):
        fate = fates.get(str(i), "unmatched") if i in unmatched_idx \
            else "matched"
        anatomy[fate] = anatomy.get(fate, 0) + 1
    assert sum(anatomy.values()) == len(gold)
    report["anatomy"] = anatomy
    return report


def render_markdown(report: dict, title: str = "Evaluation report") -> str:
    """Render the report record as a Markdown report card."""
    raw = report["raw"]
    lines = [f"# {title}", ""]
    lines.append("## Raw scores (exact match)")
    lines.append("")
    lines.append("| Metric | k | n | Rate | 95% CI |")
    lines.append("|---|---|---|---|---|")
    for name in ("precision", "recall"):
        r = raw[name]
        ci = r["wilson95"]
        ci_s = f"{ci[0]}-{ci[1]}" if ci[0] is not None else "n/a"
        lines.append(f"| {name} | {r['k']} | {r['n']} | "
                     f"{r['rate']} | {ci_s} |")
    corrected = report.get("corrected")
    if corrected:
        lines += ["", "## Audit-corrected", ""]
        if "precision" in corrected:
            lines.append(
                f"- Corrected precision: {corrected['precision']}")
        if "ceiling_adjusted_recall" in corrected:
            lines.append(
                "- Ceiling-adjusted recall: "
                f"{corrected['ceiling_adjusted_recall']}")
    anatomy = report["anatomy"]
    lines += ["", "## Failure anatomy (gold fates)", "",
              "| Fate | Count |", "|---|---|"]
    for fate in sorted(anatomy, key=lambda f: -anatomy[f]):
        lines.append(f"| {fate} | {anatomy[fate]} |")
    lines.append("")
    return "\n".join(lines)
