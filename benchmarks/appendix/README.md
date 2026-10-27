# Appendix — adjudication samples & verdicts (Re-DocRED record)

Supplementary evidence appendix to the flagship quality record
(canonical text: the [whitepaper](../../docs/whitepaper.md), §5; in-tree pointer:
[`benchmarks/redocred.md`](../redocred.md)).

> **Label — machine judgments.** Every verdict in this
> appendix is a **machine adjudication**: blind samples judged
> by machine judges under the record's pre-registered
> protocol, with verbatim-quote verification and embedded
> controls. These are not human labels, and this appendix is
> **not a corrected gold dataset** — it is the evidence behind
> the record's corrected estimates, shipped so a reader can
> re-check them.

## What is here

The judgment samples for both estimation directions, their
key files (stratum / population / control flags — withheld
from the judges), the judges' verdicts, the sampling-frame
manifest, and the raw census scores the correction starts
from. The estimator code ships as the verification cores
(`harness/verification/` — corrected precision,
ceiling-adjusted recall, stratified estimation).

Two verdict sets ship: the original adjudication
protocol's (`judgments/`) and the **ruled** set
(`judgments_ruled/` + `RULING.md`) — the 565 seam-flagged
items re-judged blind under the adjudication contract
completed by the six seam rulings of 2026-10-22
([`docs/ADJUDICATION.md`](../../docs/ADJUDICATION.md)).
The record's headlines are the ruled estimates; the
original set is the measurement's provenance.

**Reproduction.** `examples/demo/replay_scores.py` recomputes
every headline of the record from these files alone, under
both verdict sets — the ruled set, which carries the
record's headlines, and the original protocol's, kept as
provenance — together with the blind controls, the
quote-verification failures, and all five fate support
rates, and exits non-zero if any reproduced value leaves
the published figure (the figures themselves: whitepaper
§5; claim index in [`benchmarks/README.md`](../README.md)).
Verified at assembly and again when the ruled set was
added (2026-10-22): all headlines reproduce, both sets.

## Manifest — exactly what was copied

Verbatim copies from the lab run record
(`batches/redocred-mainrun-v1/`, the flagship build's batch
directory); sha256 of the staged file equals the source's.

| Staged file | Lab source | sha256 | Content |
|---|---|---|---|
| `sample_a_items.jsonl` | `s3_sample_a_items.jsonl` | `353fa23fed1bebcc121baee67c4fff05a5b4e2aea9ee46fa15ef8d60e2e85269` | Direction A items as presented to the judges (340: 300 sample + 40 blind controls), each with its document context |
| `sample_a_key.jsonl` | `s3_sample_a_key.jsonl` | `003b8a0172b12c09b3079ee579e805f9102c6d77007f5173bc9135807696d634` | Direction A key: stratum (pair_matched / no_gold_pair), population, control flag, gold-match status |
| `sample_b_items.jsonl` | `s3_sample_b_items.jsonl` | `2677bf2df619da6cb5e397f24ff60dc21da46e810426ea30594f1b3b65878338` | Direction B items (400 gold triples) with document context |
| `sample_b_key.jsonl` | `s3_sample_b_key.jsonl` | `83076cc909cec39b6a535eca94c28f1109cfac96162c9aad52de9d80c920a794` | Direction B key: corpus population, run fate per item |
| `judgments/a1.jsonl` … `a4.jsonl` | `judgments/a1.jsonl` … `a4.jsonl` | `432b7568c00b24f6…` · `99b266ba8024595d…` · `9cb7e0c315544271…` · `09faaee458ea2b0b…` (full hashes in the assembly manifest) | Direction A verdicts (85 × 4 = 340): verdict + evidence quote + note |
| `judgments/b1.jsonl` … `b4.jsonl` | `judgments/b1.jsonl` … `b4.jsonl` | `8415a14273cb82ea…` · `8878ef4f7b49816d…` · `7c74b60c3c224f5c…` · `a1d7b30444776da7…` (full hashes in the assembly manifest) | Direction B verdicts (100 × 4 = 400) |
| `sample_manifest.json` | `s3_sample_manifest.json` | `a0ccea3d5d6e100a8a22b9e3124e5fc50a57561e872173d1951c126f27bc0133` | Sampling frames, strata populations, allocations, seeds, blinding record |
| `raw_scores.json` | `s3_raw_scores.json` | `7396786e3b529b2e159b2a1fbeebcf7f9446e50773400afe0c5e85a1195b4802` | The raw census scores (FULL-500 / CONF-440) the correction starts from |

Added 2026-10-22 — the ruled verdict set, from
the adjudication recheck's verification record (the
ruled re-adjudication of 2026-10-22; account in the
whitepaper §5.4 and in `RULING.md` itself). The verdict
files are the ruled run's `ruled_verdicts.jsonl`
(sha256 `21cb4467114d5310e03b02ec319392a4d1646ea5badc7d5e934078d956af68da`)
re-serialized per direction — `queue_id` renamed to the
appendix's `id`, rows sorted by id; verdicts, evidence,
notes, and `seam_applied` values are the run's own:

| Staged file | sha256 | Content |
|---|---|---|
| `judgments_ruled/a.jsonl` | `178379e0e5943ab409b8496aced2a33dc0a5cfd8d7a41a875305a8d3f9c66ad1` | Ruled verdicts, Direction A items (240 of the 565 flagged) |
| `judgments_ruled/b.jsonl` | `adb2aaa7e20adc1b096241827af67c88642163803289a3b0c071549a4cce9194` | Ruled verdicts, Direction B items (325 of the 565 flagged) |
| `RULING.md` | `acff9f4f62bc1835deec788d648a761cae95db5a1a27e1fe02252dfef431ce24` | The ruling's provenance and the packet text verbatim (R1–R6), as the ruled judges received it |

Added 2026-10-23 — the cross-family verdicts, from
the post-ruling agreement re-measurement recorded in the
whitepaper §5.4 addendum of the same date. A third verdict
set, of a different kind: it feeds no estimate. A fresh
stratified sample of 150 direction-A items (seed
20261024) was judged blind under the ruled contract by
`anthropic/claude-sonnet-5.5` (the original panel's
family) and compared against the ruled set above:
agreement 0.7727 over the 110 comparable pairs (Cohen's
κ = 0.4271, three-category); 0.8019 over the 106 decisive
pairs (κ = 0.4723). Verbatim copies from the run record —
the runner is the run's own script, transport fixes and
all:

| Staged file | sha256 | Content |
|---|---|---|
| `judgments_e6/e6_sample.json` | `8e782c314de8b1fa4646913a6b6565704f0dfa56eb165862a8606e14d5a4bced` | The frozen sample: seed, strata, and the 150 sampled item ids |
| `judgments_e6/e6_verdicts.jsonl` | `592b247f3c546cf10fd95f1bf47667bcbcc60e2f56ba6b508f6db68f11b4b307` | The fresh judge's 150 verdicts with per-call cost |
| `judgments_e6/e6_results.json` | `46affa1042833c38748782262d481adfff88d1dd18205e56e10476308b612582` | Agreement + κ under both conventions, decisive-pair confusion, run spend ($1.0443) |
| `judgments_e6/run_e6.py` | `7b4fd248e69c4dbac7aae460232a2b481f71c0743e143f085dfb0c4372c6f5a2` | The runner as executed: sample construction, blind packets, quote verification, both κ conventions, spend cap |

Pre-copy sweep (at assembly): every copied file was
scanned for credential shapes, personal data, and absolute
lab paths — zero hits. Two provenance strings remain, by
design: `sample_manifest.json` names the lab design document
by its repo-relative path (its public distillation is the
quality record — [whitepaper](../../docs/whitepaper.md) §5; in-tree pointer
`benchmarks/redocred.md`), and item contexts are benchmark
document text (below).

## Withheld from the appendix — recorded, not copied

| Lab artefact | Lab sha256 | Why it stays in the lab |
|---|---|---|
| `s3_corrected_scores.json` (the aggregator's output file) | `1e22acc034cc2b686cb836c1307d37e3bf2de9c6642a61196b378020f9298b17` | It embeds the program-spend composition (per-cell spend detail and the program total). The project's cost rule publishes the reference run's aggregate only; the full ledger and program totals are internal provenance. Every score the record cites from this file is published in the [whitepaper](../../docs/whitepaper.md) (§5), and the replay above reproduces them from the shipped inputs — the withheld file is not needed to check the record. |
| `sample_manifest.jsonl` (the Direction B sampling frame) | `02aefa83646e0c7a77dbbfef6c0d9f32153a4067133c344360edfdaca5591cbf` | The frame carries the **full text of all 500 dev documents** — corpus text, which this project fetches from source and never bundles. The frame's definitions (populations, strata, seeds) ship in `sample_manifest.json`; the sampled items' own contexts ship in the sample files. |

## Upstream attribution

The sample items carry document-context excerpts from
**Re-DocRED** (Tan et al., *Revisiting DocRED*, EMNLP 2022),
shipped here as adjudication evidence. Re-DocRED is
distributed under the MIT license; its document text derives
from Wikipedia and remains under the upstream CC BY-SA
terms. The corpus itself is **not** redistributed by this
repository — the demo fetch path downloads it from its
source (`docs/hands-on-appendix.md`).
