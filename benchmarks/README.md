# Benchmarks — the quality records and their data

The quality records' canonical text lives in the
[**whitepaper**](../docs/whitepaper.md) (placement revision,
2026-10-22): the scores travel with their scope, method,
and failure anatomy in the long-form document, not on the
repository's front surfaces. What ships here is the
records' index, their evidence data, and the replay
tooling — everything a reader needs to check the
whitepaper's numbers from this tree.

Corrected numbers are **machine-adjudicated** wherever
they appear (see each record's label in the whitepaper).

## The records

This index is the one figure surface besides the
whitepaper: other documents point here (or to the
whitepaper) rather than restating the figures.

| Record | Scope | Headline | Full text |
|---|---|---|---|
| Flagship build — Re-DocRED (Open Domain pack) | Re-DocRED dev, all 500 documents; pinned reference profile; run 2026-10-21 | corrected precision **0.9110** [0.8919–0.9275]; ceiling-adjusted recall **0.2028**; one pre-registered criterion (C-2) **FAIL**, recorded as failed | [whitepaper](../docs/whitepaper.md) §5.1–§5.6; in-tree pointer + anatomy data: [`redocred.md`](redocred.md) |
| BioRED pilot (Bio Medical pack) | BioRED dev, 100 documents; schema-first profile; test split sealed | recall 0.5120 PASS; audit-corrected precision **0.9510**; frozen grounded precision 0.6625 **FAIL** beside it | [whitepaper](../docs/whitepaper.md) §5.7; in-tree pointer: [`biored-cuad.md`](biored-cuad.md) |
| CUAD certification (Legal Contract pack) | CUAD v1, full corpus (510 documents); production retrieval path | promotion verdict **PROMOTABLE**; certification audit precision 0.7185; pinned-QA recall@20 graph+hybrid **0.7115** (long documents) | [whitepaper](../docs/whitepaper.md) §5.8; in-tree pointer: [`biored-cuad.md`](biored-cuad.md) |
| Pack conformance (all five packs) | per-entry: gate, instrument, population, delta | 22 registered entries — adoptions, measured rejections, and closures, negatives included | `packs/*/conformance.json`; indexed below |

## Datasets

The three datasets the records measure on — none is
bundled in this repository (the appendix exception is
stated in the [NOTICE](../NOTICE)); each is fetched
from its source.

| Dataset | Records | Source | Paper | License / terms |
|---|---|---|---|---|
| Re-DocRED | Flagship build (Open Domain pack) — dev split, all 500 documents | [github.com/tonytan48/Re-DocRED](https://github.com/tonytan48/Re-DocRED) | Tan, Xu, Bing, Ng & Aljunied, *Revisiting DocRED – Addressing the False Negative Problem in Relation Extraction*, EMNLP 2022 | Dataset MIT; document text derives from Wikipedia and remains CC BY-SA |
| BioRED | Bio Medical pilot — dev split, 100 documents | [ftp.ncbi.nlm.nih.gov/pub/lu/BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/) | Luo, Lai, Wei, Arighi & Lu, *BioRED: a rich biomedical relation extraction dataset*, Briefings in Bioinformatics, 2022 | Distributed by NCBI under its own terms (cite the authors) |
| CUAD v1 | Legal Contract certification — full corpus, 510 documents | [github.com/TheAtticusProject/cuad](https://github.com/TheAtticusProject/cuad) (The Atticus Project) | Hendrycks, Burns, Chen & Ball, *CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review*, NeurIPS 2021 | CC BY 4.0 |

Redistribution terms for every third-party source are
the [NOTICE](../NOTICE).

## Evidence and replay

- [`appendix/`](appendix/) — the flagship record's
  adjudication samples, key files, and verdicts (both
  sets: the original protocol's and the ruled
  re-adjudication's, with the ruling text), labeled
  machine judgments, with the sampling manifest
  and the raw census scores.
- `examples/demo/replay_scores.py` — recomputes every
  flagship headline from the appendix alone, under both
  verdict sets (the ruled set carries the record's
  headlines); exits non-zero on drift.
- `harness/verification/` — the estimator and audit cores
  the records were computed with (corrected precision,
  ceiling-adjusted recall, stratified estimation).
- `assets/anatomy.png` — the flagship failure-anatomy
  chart, rendered by `harness/render_charts.py` from the
  record data block in [`redocred.md`](redocred.md); a
  whitepaper figure (the README carries no score charts).

## Claim index — claim → harness → readout

Every published number, one line each: the headline
figure, its scope, the shipped harness that produces or
checks it, and the readout where it lives.

### Flagship — Re-DocRED build (Open Domain)

| Claim | Headline | Scope | Harness | Readout |
|---|---|---|---|---|
| Raw benchmark precision | 0.5723 | FULL-500 census, frozen scorer | `harness/verification` scoring (`verify score`) | whitepaper §5.2; appendix `raw_scores.json` |
| Corrected precision | **0.9110** [0.8919–0.9275] | FULL-500, sample estimate, ruled adjudication contract | verification scoring + audit cores; replay `examples/demo/replay_scores.py` (computes the original-protocol estimate, 0.9765, beside it as provenance) | whitepaper §5.2; appendix samples + both verdict sets |
| Ceiling-adjusted recall | **0.2028** [0.1856–0.2237] | FULL-500, sample estimate, ruled adjudication contract | same replay | whitepaper §5.2 |
| Failure anatomy | 57.9% of gold never enumerated | 17,284 gold, census fates + sampled support rates | verification report core (failure anatomy) | whitepaper §5.3; data block in `redocred.md` |
| Criterion C-2 (NONE-loss share) | 0.5128 vs ≤ 0.50 bar — **FAIL** | CONF-440, pre-registered | verification config (gate definitions) | whitepaper §5.5 |
| JF-F1 direction slot — adopted | η 0.1124 → 0.2596; flips 131 → 7 | judge-fitting cell | verification cell runner | whitepaper §5.5; pack entry DC-3 / GC-3 |
| J3F2 document-context judge — rejected | η 0.3071 vs gate 0.3096 (≈ 2 gold pairs) | pre-registered cell | verification cell runner | whitepaper §5.5 |
| Enumeration operating point | 52,463 pairs; est. recall 0.1884 (adoption-time projection, not recomputed under the ruled re-price) | dev-500 projection at realized rate | verification cell runner + counted projection (`verify project`) | whitepaper §5.6; pack entry GC-4 |
| Reference-run cost (aggregate) | $18.7980 key-observed total (ledger $18.6802); candidate judging $0.0006683/pair | 500 docs, 26,938 pairs, 2026-10-21 | spend ledger + counted projection | whitepaper §5.6 |
| Shipped readout | Tier-1 **4,088** (3,902 + 186 supplementary promotion) | frozen estimates not recomputed over the promoted set | pipeline batch + gates | whitepaper §5.1 |

### Bio Medical — BioRED pilot (Dev-100)

| Claim | Headline | Scope | Harness | Readout |
|---|---|---|---|---|
| Recall | 0.5120 — PASS (bar ≥ 0.35) | Dev-100 census, frozen scorer | pipeline batch + frozen scorer semantics | whitepaper §5.7 |
| Association recall | 0.4768 — PASS (bar ≥ 0.30) | Dev-100 census | same | whitepaper §5.7 |
| Grounded precision, frozen reading | 0.6625 — **FAIL** (bar ≥ 0.70) | Dev-100 census | same | whitepaper §5.7 |
| Audit-corrected grounded precision | **0.9510** [0.9398–0.9623] — PASS | Stage D audit, P2 pattern | verification audit + scoring cores | whitepaper §5.7 |
| Audit-corrected Tier-1-wide precision | **0.9321** | Stage D audit — all 2,058 Tier-1 rows, grounded and ungrounded together | verification audit + scoring cores | whitepaper §5.7; record data — see [`reproduce/README.md`](../reproduce/README.md) |
| Conversion | η_dev 0.7475 (595 of 796 reachable) | Dev-100 census | verification report core | whitepaper §5.7 |
| Schema-first effect (CR-1) | grounding 14.8% → 74.6%; recall 0.0620 → 0.2091 | round-2 Dev measurement | setup service / profile channel | whitepaper §5.7; pack `bio-medical` conformance CR-1 |
| J3 judge (CR-2) | η 0.5412 → 0.7412; NONE losses 25 → 4 | frozen slice census | verification cell runner | whitepaper §5.7; pack conformance CR-2 |
| Round-2 Tier-1 precision (P2 audit) | 0.9858 | machine-adjudicated audit | verification audit core | whitepaper §5.7 |
| Relation typing (CR-3…CR-7) | five candidates measured, **none adopted** | slice + one registered Dev evaluation | verification cell runner | whitepaper §5.7; pack conformance CR-3…CR-7 |

### Legal Contract — CUAD

| Claim | Headline | Scope | Harness | Readout |
|---|---|---|---|---|
| Extraction bake-off (CC-2) | presence precision 0.886 → **0.963** (Haiku seat adopted) | 80 contracts, span-gated | clause extraction harness + span gate | whitepaper §5.8; pack conformance CC-2 |
| Full-corpus clause run | precision 0.938 / recall 0.840 | CUAD 510 docs | same | whitepaper §5.8 |
| Certification audit (CC-3) | precision 0.7185 (140 sampled, 135 decided) | served lineage, blind labels | promotion gate (audit leg) | whitepaper §5.8 |
| Pinned QA — graph | recall@20 0.6516 | 116 questions, production path | query stack (R16 path) | whitepaper §5.8 |
| Pinned QA — hybrid | recall@20 0.6314 | same QA, same corpus | query stack | whitepaper §5.8 |
| Pinned QA — graph+hybrid | recall@20 **0.7115** (+0.080 over hybrid) | long documents — stated scope | query stack | whitepaper §5.8 |
| Promotion verdict (CC-3) | PROMOTABLE (gate PASS × 3) | 2026-10-16 final re-run | promotion gate | whitepaper §5.8; pack conformance CC-3 |
| Profile approval (CC-1) | 16/16 decision points approved | CUAD census, 510 docs | setup service census + scanners | whitepaper §5.8; pack conformance CC-1 |
| Framing as data (CC-4) | prompts byte-identical after PK-01 move | proposer + slot surfaces | profile config (payload channel) | pack conformance CC-4 |

## Pack conformance entries (all packs)

One line per entry; the full entry (gate, instrument,
population, delta) lives in each pack's `conformance.json`.

| Pack | Entry | Status | Headline |
|---|---|---|---|
| Bio Medical | CR-1 | adopted | schema-first profile v2: grounding 14.8% → 74.6% |
| Bio Medical | CR-2 | adopted | J3 judge v4: η 0.5412 → 0.7412 on the slice census |
| Bio Medical | CR-3 | measured, not adopted | type guidance T1: agreement −0.0635 vs control |
| Bio Medical | CR-4 | measured, not adopted | type-evidence T3: agreement −0.079; precision leg broken |
| Bio Medical | CR-5 | measured, not adopted | Association specialist T2a: Association agreement 24/30 → 6/30 |
| Bio Medical | CR-6 | measured, not adopted | definitions-free second look T2b: net agreement +1 |
| Bio Medical | CR-7 | closed | Train-fitted picker T4: Dev agreements 390 → 358 (leg ≥ +30) |
| Legal Contract | CC-1 | adopted | profile approval 2026-10-16, 16 decision points |
| Legal Contract | CC-2 | adopted | extraction bake-off: Haiku seat, precision 0.963 |
| Legal Contract | CC-3 | adopted | certification PROMOTABLE: audit 0.7185; QA 0.7115 |
| Legal Contract | CC-4 | adopted | PK-01: domain framing moved into payload data, prompts byte-identical |
| Open Domain | DC-1 | measured baseline | profile v1: raw P 0.544 vs audit-corrected 0.987 (30-doc audit) |
| Open Domain | DC-2 | measured pilot | pack candidate: corrected P 0.9818 → 0.9869 (v1 → v2 payload) |
| Open Domain | DC-3 | measured, adopted | JF-F1 cell: η 0.1124 → 0.2596; candidate P 0.3464 → 0.8381 |
| Open Domain | DC-4 | measured, not adopted | J5s0 subset cell: η 0.2526 vs F1 0.2596; NONE-loss 409 vs 258 |
| Open Domain | DC-5 | measured, adopted | roster-first enumeration (RS-01): endpoint-never-proposed losses 809 → 426 |
| FIBO Contracts | OB-1 | emitted (unmeasured) | ontology import: FIBO FND Agreements → 15 entity + 13 relation types; explicitly not a measured claim |
| Core | GC-1 | adopted | core gate/ER defaults carried as baseline payload data |
| Core | GC-2 | adopted | neutral core framing as payload data (PK-01) |
| Core | GC-3 | adopted | JF-F1 direction slot as core mechanism (η 0.1124 → 0.2596) |
| Core | GC-4 | adopted | Enumeration operating point: max_dist=1 + REC (52,463 pairs, $35.06 projected) |
| Core | GC-5 | adopted | reference profile = connection-layer preset; packs carry no model names |

## Pack auxiliary documents

Two packs carry human-readable auxiliary documents
(optional, non-normative — see the pack contract,
[`docs/PACK_DESIGN.md`](../docs/PACK_DESIGN.md) §1.1):

- [`packs/bio-medical/profile/CENSUS.md`](../packs/bio-medical/profile/CENSUS.md)
  — the Bio Medical pack's corpus census: a direct-parse
  inventory of the BioRED corpus the pack was fitted on.
- [`packs/legal-contract/profile/profile.md`](../packs/legal-contract/profile/profile.md)
  — the Legal Contract pack's corpus profile: the
  stage-0 setup-service profile of the CUAD corpus.

That is the whole index: every record above can be
recomputed or re-read from the files it names.
