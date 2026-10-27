# Quality record — Re-DocRED flagship build (pointer)

**The record's canonical text lives in the [whitepaper](../docs/whitepaper.md),
§5** (placement revision, 2026-10-22: the
scores moved out of the repository's front surfaces into
the long-form document, where they travel with their
scope, method, and failure anatomy). This file is the
record's in-tree pointer: its scope, its replay path, and
the anatomy data block the chart renderer parses. The
claim-level index of every published number is
[`benchmarks/README.md`](README.md).

## Scope

One measured build, frozen as measured: the
Open Domain pack's flagship build over the
Re-DocRED dev split (all 500 documents), executed
2026-10-21 under the pinned reference profile. The
whitepaper record (§5.1–§5.6) carries the build
description, the raw and corrected headline scores with
their intervals, the full failure anatomy, the method,
the pre-registered criteria — including the criterion
that **failed**, recorded as a failure — the gate
adoptions and rejections, the shipped operating point,
and the reference run's aggregate cost.

Dataset: **Re-DocRED** — Tan et al., *Revisiting
DocRED – Addressing the False Negative Problem in
Relation Extraction* (EMNLP 2022). Source:
[github.com/tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED)
(dataset MIT; document text CC BY-SA — terms in the
[NOTICE](../NOTICE)).

> **Label.** Every corrected number in the record rests
> on **machine adjudication**: blind samples judged by
> machine judges under the adjudication contract
> ([`docs/ADJUDICATION.md`](../docs/ADJUDICATION.md) —
> the as-typed contract completed by its six seam
> rulings), with verbatim-quote verification and
> embedded controls.
> They are machine judgments, not a human audit, and
> they are not a corrected gold dataset.

## Replay — check the record from the shipped data

The adjudication samples, their key files, and the
verdicts ship as the supplementary appendix
([`appendix/`](appendix/)), labeled machine judgments —
evidence a reader can re-check, not a standalone
dataset. Both verdict sets ship (the original protocol's
and the ruled re-adjudication's, with the ruling text);
`examples/demo/replay_scores.py` recomputes every
headline of the record from the appendix files alone —
both estimates — and exits non-zero on drift; the
estimation code ships as the verification cores
(`harness/verification/`).

## Record data — failure anatomy (parsed, never retyped)

The two tables below are the record's data block:
`harness/render_charts.py` parses them to render the
anatomy chart (`assets/anatomy.png`, a whitepaper
figure), and `tests/test_charts.py` pins the parse
against the frozen values. They are carried verbatim
from the record (whitepaper §5.3).

Every one of the 17,284 gold triples has exactly one fate:

| Gold fate | Count | What it means |
|---|---|---|
| matched | 1,838 | Emitted in Tier-1 exactly |
| emission_blocked | 1,759 | A covering pair was judged OK, but no Tier-1 exact match was emitted |
| none_judged | 3,677 | The pair was enumerated; the judge answered NONE |
| never_enumerated | 6,340 | Both endpoints rostered, but never co-occurring inside the sentence window |
| roster_missed | 3,670 | An endpoint never made the roster |

Text-support rates by fate (machine-adjudicated under
the ruled contract, decisive items):

| Gold fate | Sampled | Text-supported rate |
|---|---|---|
| matched | 44 | 0.884 [0.755–0.949] |
| none_judged | 86 | 0.518 [0.413–0.621] |
| emission_blocked | 32 | 0.500 [0.336–0.664] |
| never_enumerated | 153 | 0.424 [0.346–0.505] |
| roster_missed | 85 | 0.529 [0.424–0.632] |
