"""Render the README charts from the shipped quality record.

The numbers plotted here are PARSED from the record data
block in ``benchmarks/redocred.md`` — never retyped. (The
record's canonical text lives in the whitepaper, §5;
the benchmarks file
carries the anatomy tables as data for this renderer and
the replay.) A chart whose numbers disagree with the
record is a defect, so parsing fails closed: if the
anatomy tables drift from the shape this renderer expects
(five fates, counts summing to the stated gold total),
rendering refuses instead of plotting something
unverifiable.

Outputs:
  * ``benchmarks/assets/anatomy.png`` — the flagship failure
    anatomy: census counts per gold fate + the adjudicated
    text-support rate per fate (CI whiskers: the rates are
    sample estimates). A whitepaper asset: the README
    carries no score charts.
  * ``docs/assets/method.png`` — the method diagram:
    proposer/judge separation across the build loop.
  * ``pipeline-flow.png`` — the batch stage contract
    (INGEST → … → VERIFY, from ``pipeline/batch_cli.py``)
    with the ROUTE tier branches as coded in
    ``pipeline/batch_run.py`` (calibration from the shipped
    record: threshold 0.25, review band 0.25–0.50).
  * ``ontology-loop.png`` — the ontology builder loop
    (draft → validate → approve → emit) with its refusal
    paths, from ``pipeline/ontology_builder.py``.

The diagrams render the product's actual structure — stage
names, branch conditions, and refusal paths are taken from
the shipped code and record, never drawn from imagination.
All four figures are re-renderable at other sizes for the
whitepaper.

Style follows the repo chart rules: one deep-navy palette, value
labels on bars, CI whiskers wherever a number is an
estimate, the source named in every figure.

Usage: ``python -m harness.render_charts [--out-dir DIR]``
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_mpl_cache = None


def _mpl():
    """matplotlib, imported on first use (and cached).

    Parsing the record (parse_anatomy) is dependency-free —
    only rendering needs matplotlib, which is the [viz]
    extra, not a core dependency: a base install must import
    this module and run the parse tests."""
    global _mpl_cache
    if _mpl_cache is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        _mpl_cache = (plt, FancyArrowPatch, FancyBboxPatch)
    return _mpl_cache


ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "benchmarks" / "redocred.md"

NAVY = "#12355b"
NAVY_LIGHT = "#5b84ae"
ACCENT = "#c05621"
GREY = "#6b7280"
# Method-diagram emphasis: the proposer proposes and
# the judge decides, so the judge carries the row's strongest
# weight — full NAVY. The proposer keeps a warm family tint
# desaturated to a quiet clay, and Graph sits one step below
# the judge in a mid-navy.
CLAY = "#b8876e"
NAVY_MID = "#2c5178"

# Display order: the matched mass first, then loss strata by
# size. Keys are the record's own fate names.
FATE_ORDER = ["matched", "never_enumerated", "roster_missed",
              "none_judged", "emission_blocked"]
FATE_LABELS = {
    "matched": "matched",
    "never_enumerated": "never enumerated",
    "roster_missed": "roster missed",
    "none_judged": "judge said NONE",
    "emission_blocked": "blocked at emission",
}

# Census rows carry prose in the third cell; the support-rate
# table's rows carry a number there and must not match this.
_COUNT_ROW = re.compile(r"^\| (\w+) \| ([\d,]+) \| [^0-9\[]")
_RATE_ROW = re.compile(
    r"^\| (\w+) \| (\d+) \| ([\d.]+) \[([\d.]+)[\u2013-]([\d.]+)\] \|")
_TOTAL = re.compile(r"Every one of the ([\d,]+) gold triples")


def parse_anatomy(record_text: str) -> dict:
    """Parse the failure-anatomy tables from the record.

    Returns {total, fates: {fate: {count, sampled, rate,
    ci}}}. Raises ValueError on any drift from the expected
    shape — the renderer never guesses.
    """
    counts: dict[str, int] = {}
    rates: dict[str, dict] = {}
    total = None
    for line in record_text.splitlines():
        m = _TOTAL.search(line)
        if m:
            total = int(m.group(1).replace(",", ""))
        m = _COUNT_ROW.match(line)
        if m and m.group(1) in FATE_LABELS and m.group(1) not in counts:
            counts[m.group(1)] = int(m.group(2).replace(",", ""))
        m = _RATE_ROW.match(line)
        if m and m.group(1) in FATE_LABELS:
            rates[m.group(1)] = {
                "sampled": int(m.group(2)), "rate": float(m.group(3)),
                "ci": [float(m.group(4)), float(m.group(5))]}
    if total is None or set(counts) != set(FATE_LABELS) \
            or set(rates) != set(FATE_LABELS):
        raise ValueError("record anatomy tables not found in expected shape")
    if sum(counts.values()) != total:
        raise ValueError(
            f"anatomy counts sum {sum(counts.values())} != stated total {total}")
    return {"total": total,
            "fates": {f: {"count": counts[f], **rates[f]} for f in FATE_ORDER}}


def render_anatomy(anatomy: dict, out: Path) -> Path:
    """Render the two-panel anatomy chart (counts + rates)."""
    plt = _mpl()[0]
    fates = anatomy["fates"]
    labels = [FATE_LABELS[f] for f in FATE_ORDER]
    counts = [fates[f]["count"] for f in FATE_ORDER]
    rates = [fates[f]["rate"] for f in FATE_ORDER]
    colors = [ACCENT] + [NAVY] * (len(FATE_ORDER) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    bars = ax1.barh(labels[::-1], counts[::-1], color=colors[::-1], height=0.62)
    for bar, value in zip(bars, counts[::-1], strict=True):
        ax1.text(bar.get_width() + 90, bar.get_y() + bar.get_height() / 2,
                 f"{value:,}", va="center", fontsize=9, color=NAVY)
    ax1.set_xlim(0, max(counts) * 1.22)
    ax1.set_xlabel(f"gold triples (of {anatomy['total']:,}; census)", fontsize=9)
    ax1.set_title("Where the gold went — fate of every gold triple",
                  fontsize=10.5, color=NAVY, loc="left", weight="bold")
    ax1.tick_params(labelsize=9)
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    ax2.barh(labels[::-1], rates[::-1], color=NAVY_LIGHT, height=0.62)
    # One errorbar per row, at the bar's own y position (the
    # category order matches the left panel: FATE_ORDER runs
    # top to bottom). Per-row calls keep rate/CI pairing
    # explicit — no reversal arithmetic to get wrong.
    for idx, fate in enumerate(FATE_ORDER):
        y = len(FATE_ORDER) - 1 - idx
        d = fates[fate]
        ax2.errorbar(d["rate"], y,
                     xerr=[[d["rate"] - d["ci"][0]],
                           [d["ci"][1] - d["rate"]]],
                     fmt="none", ecolor=NAVY, elinewidth=1.1, capsize=3)
        # Label beyond the CI whisker's upper end — never on it.
        ax2.text(d["ci"][1] + 0.02, y, f"{d['rate']:.3f}",
                 va="center", fontsize=9, color=NAVY)
    ax2.set_xlim(0, 1.34)
    ax2.set_xlabel("text-supported rate (machine-adjudicated sample)", fontsize=9)
    ax2.set_title("…and most of it was findable",
                  fontsize=10.5, color=NAVY, loc="left", weight="bold")
    ax2.tick_params(labelsize=9)
    ax2.set_yticklabels([])
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.text(0.995, 0.01, "Source: quality record (whitepaper §5) · data: "
             "benchmarks/redocred.md · appendix: benchmarks/appendix/",
             ha="right", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def render_method(out: Path) -> Path:
    """Render the method diagram (proposer/judge separation).

    One story: how a graph is built and served. Verify is not
    drawn here — it lives in-flow as stage 8 of the pipeline
    diagram, and the measurement discipline appears only as
    the footer line. The serving surface is drawn,
    not annotated: the graph projects into the serving store
    and the store feeds its three faces (docs/serving.md).
    """
    plt, FancyArrowPatch, FancyBboxPatch = _mpl()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 34)
    ax.axis("off")

    boxes = [
        (1, 20, 13, 9, "Corpus", "your documents", NAVY_LIGHT),
        (16.5, 20, 14, 9, "Profile", "deterministic census", NAVY_LIGHT),
        (33, 20, 15, 9, "Ontology pack", "schema + definitions\nhuman-approved", NAVY_LIGHT),
        (50.5, 20, 15, 9, "Proposer", "extracts candidates", CLAY),
        (68, 20, 16, 9, "Jev gates + judge", "decide what is true\nenough to write", NAVY),
        (86.5, 20, 17, 9, "Graph", "Tier-1 triples\nevidence + provenance", NAVY_MID),
    ]
    for x, y, w, h, title, sub, color in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor=color, edgecolor="none"))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
                fontsize=9.5, weight="bold", color="white")
        ax.text(x + w / 2, y + h - 6.1, sub, ha="center", va="center",
                fontsize=7.3, color="white")
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + boxes[i][2] + 0.4
        x1 = boxes[i + 1][0] - 0.4
        ax.add_patch(FancyArrowPatch((x0, 24.5), (x1, 24.5),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=GREY, lw=1.2))
    ax.text(58, 31.5, "proposes — never decides", fontsize=8.5,
            color=CLAY, ha="center", style="italic")
    ax.text(76, 31.5, "decides — never proposes", fontsize=8.5,
            color=NAVY, ha="center", style="italic")

    # Serving surface — solid flow, drawn elements throughout.
    # Graph -> serving store (the projection), then a bus that
    # drops into the surface's three faces.
    ax.add_patch(FancyArrowPatch((95, 19.6), (95, 18.3),
                                 arrowstyle="-|>", mutation_scale=11,
                                 color=GREY, lw=1.2))
    ax.add_patch(FancyBboxPatch(
        (86.5, 13.6), 17, 4.4, boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor="white", edgecolor=NAVY, lw=1.1))
    ax.text(95, 16.4, "Serving store", ha="center", va="center",
            fontsize=8.5, weight="bold", color=NAVY)
    ax.text(95, 14.6, "Neo4j projection", ha="center", va="center",
            fontsize=6.9, color=NAVY)
    ax.plot([95, 95], [13.6, 12.6], color=GREY, lw=1.2)
    ax.plot([61, 95], [12.6, 12.6], color=GREY, lw=1.2)
    faces = [
        (53.25, "Search", "lexical · graph arms"),
        (70.25, "API", "sample HTTP API"),
        (87.25, "App", "fact + evidence view"),
    ]
    for x, title, sub in faces:
        cx = x + 7.75
        ax.add_patch(FancyArrowPatch((cx, 12.6), (cx, 11.9),
                                     arrowstyle="-|>", mutation_scale=11,
                                     color=GREY, lw=1.2))
        ax.add_patch(FancyBboxPatch(
            (x, 6.4), 15.5, 5.2, boxstyle="round,pad=0.4,rounding_size=1.2",
            facecolor="white", edgecolor=NAVY, lw=1.1))
        ax.text(cx, 9.9, title, ha="center", va="center",
                fontsize=9, weight="bold", color=NAVY)
        ax.text(cx, 7.7, sub, ha="center", va="center",
                fontsize=6.9, color=NAVY)
    ax.text(52.5, 1.6, "every build is measured; configuration changes pass "
            "pre-registered gates or stay off", ha="center", fontsize=8,
            color=GREY, style="italic")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def _box(ax, x, y, w, h, title, sub, face, edge="none", lw=1.0,
         title_color="white", sub_color="white"):
    """One rounded stage box with a bold title + small subtitle."""
    FancyBboxPatch = _mpl()[2]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
        facecolor=face, edgecolor=edge, lw=lw))
    ax.text(x + w / 2, y + h - 2.5, title, ha="center", va="center",
            fontsize=9.5, weight="bold", color=title_color)
    if sub:
        ax.text(x + w / 2, y + h - 6.4, sub, ha="center", va="center",
                fontsize=7.2, color=sub_color)


def _arrow(ax, x0, y0, x1, y1, color=GREY, style="-|>", rad=0.0, lw=1.2):
    """One directed edge."""
    FancyArrowPatch = _mpl()[1]
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=11,
        color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


def render_pipeline_flow(out: Path) -> Path:
    """The 8-stage batch contract + the ROUTE tier branches.

    Stage names/order are the shipped stage contract
    (pipeline/batch_cli.py docstring); the branches are the
    ROUTE logic of pipeline/batch_run.py with the shipped
    calibration (threshold 0.25, review band 0.25–0.50).
    """
    plt = _mpl()[0]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 105)
    ax.set_ylim(8, 56)
    ax.axis("off")

    stages = [
        ("INGEST", "documents +\nprofile frozen"),
        ("NER", "contract rows only\n(local NER retired)"),
        ("CANDIDATES", "LLM proposer\n(the only source)"),
        ("GATES", "Jev gates + judge\np_true score"),
        ("ROUTE", "tier by p_true\n(branches below)"),
        ("WRITE", "Tier-1 → registry\nsource of truth"),
        ("INDEX", "Neo4j projection\n+ search indexes"),
        ("VERIFY", "scores + failure\nanatomy report"),
    ]
    w, h, gap, y = 11.2, 10, 1.55, 44
    xs = [1 + i * (w + gap) for i in range(len(stages))]
    for x, (title, sub) in zip(xs, stages, strict=True):
        face = NAVY if title in ("GATES", "ROUTE") else NAVY_LIGHT
        if title == "ROUTE":
            face = ACCENT
        _box(ax, x, y, w, h, title, sub, face)
    for i in range(len(stages) - 1):
        _arrow(ax, xs[i] + w + 0.3, y + h / 2, xs[i + 1] - 0.3, y + h / 2)

    # ROUTE branches (batch_run.py): p >= band_hi -> Tier 1;
    # threshold <= p < band_hi -> Tier 2 review queue;
    # p < threshold -> Tier 0 dropped queue (recorded).
    route_cx = xs[4] + w / 2
    lanes = [
        (6, "p ≥ 0.50", "Tier 1 · VERIFIED", "written at WRITE", NAVY),
        (38, "0.25 ≤ p < 0.50", "Tier 2 · review band", "review queue — human decides", ACCENT),
        (70, "p < 0.25", "Tier 0", "dropped queue — recorded, reviewable", GREY),
    ]
    for lx, cond, tier, dest, color in lanes:
        _arrow(ax, route_cx, y - 0.4, lx + 14, 30.8, color=color, rad=-0.12)
        _box(ax, lx, 20, 28, 10, tier, dest, "white", edge=color, lw=1.3,
             title_color=color, sub_color=color)
        ax.text(lx + 14, 33.4, cond, ha="center", fontsize=8.5,
                color=color, weight="bold")
    ax.text(53, 15.8, "guard demotions (non-asserted / whitelist) move a "
            "Tier-1 row to Tier 2 before writing",
            ha="center", fontsize=8, color=GREY, style="italic")
    ax.text(53, 11.8, "per-stage state persists under the batch — "
            "a failed run resumes, never re-spends",
            ha="center", fontsize=8, color=GREY, style="italic")
    fig.text(0.995, 0.012, "Source: pipeline/batch_cli.py stage contract · "
             "pipeline/batch_run.py ROUTE · calibration: whitepaper §5.1",
             ha="right", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def render_ontology_loop(out: Path) -> Path:
    """The ontology builder loop and its refusal paths.

    Mirrors pipeline/ontology_builder.py: validate is
    fail-closed with named findings; approval is a human
    decision bound to the schema hash; emission refuses
    invalid / unapproved / stale drafts and load-verifies
    the emitted pack before reporting success.
    """
    plt = _mpl()[0]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 44)
    ax.axis("off")

    boxes = [
        (1, 26, 21, 11, "DRAFT",
         "corpus profile + brief\ndrafting tier — or a\n"
         "deterministic skeleton", NAVY_LIGHT),
        (29, 26, 21, 11, "VALIDATE",
         "deterministic checks\nnamed findings\nfail-closed", NAVY),
        (57, 26, 21, 11, "APPROVE",
         "human decision point\napproval bound to\nthe schema hash", ACCENT),
        (85, 26, 19, 11, "EMIT", "pack written +\nload-verified", NAVY),
    ]
    for x, y, w, h, title, sub, face in boxes:
        _box(ax, x, y, w, h, title, sub, face)
    for i in range(len(boxes) - 1):
        _arrow(ax, boxes[i][0] + boxes[i][2] + 0.3, 31.5,
               boxes[i + 1][0] - 0.3, 31.5)
    # Refusal loops back to DRAFT. The validate arc stays
    # shallow; the approve arc runs deeper (longer span)
    # so the two never cross, and both land on the DRAFT
    # box. Sign convention (probe-verified): for a
    # right-to-left arc3 arrow, a NEGATIVE rad bows the
    # arc downward; positive bows it up through the boxes.
    _arrow(ax, 39.5, 25.6, 15, 25.6, color=ACCENT, rad=-0.15)
    ax.text(30, 22.0, "findings remain → revise the draft",
            ha="center", fontsize=8.5, color=ACCENT, style="italic")
    _arrow(ax, 67.5, 25.6, 8, 25.6, color=ACCENT, rad=-0.22)
    ax.text(44, 15.2, "declined — or the draft changed (stale hash) → back to draft",
            ha="center", fontsize=8.5, color=ACCENT, style="italic")
    _box(ax, 85, 6, 19, 10, "PACK", "emitted_unmeasured\nuntil a registered\nmeasurement adopts it",
         "white", edge=NAVY, lw=1.3, title_color=NAVY, sub_color=NAVY)
    _arrow(ax, 94.5, 25.6, 94.5, 16.4, color=NAVY)
    ax.text(53, 2.2, "emission refuses an invalid draft, a missing approval, "
            "or another pack's approval",
            ha="center", fontsize=8, color=GREY, style="italic")
    fig.text(0.995, 0.012, "Source: pipeline/ontology_builder.py",
             ha="right", fontsize=7.5, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    """Render all figures from the shipped record + code.

    Placement (revised 2026-10-22): the process
    diagrams go to docs/assets/ for the README; the
    quality-number chart is a whitepaper asset and goes to
    benchmarks/assets/ — the README carries no score
    charts.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(ROOT / "docs" / "assets"))
    parser.add_argument("--bench-dir",
                        default=str(ROOT / "benchmarks" / "assets"))
    parser.add_argument("--record", default=str(RECORD))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    bench_dir = Path(args.bench_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_dir.mkdir(parents=True, exist_ok=True)
    anatomy = parse_anatomy(Path(args.record).read_text(encoding="utf-8"))
    print(render_anatomy(anatomy, bench_dir / "anatomy.png"))
    print(render_method(out_dir / "method.png"))
    print(render_pipeline_flow(out_dir / "pipeline-flow.png"))
    print(render_ontology_loop(out_dir / "ontology-loop.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
