# Contributing to OntoSmith

Thanks for your interest. OntoSmith is a measured system:
its claims are numbers with instruments behind them, and
contributions are held to the same standard as the code —
if a change alters behavior, the evidence that it didn't
break anything ships with the change.

## Setup

```bash
git clone https://github.com/Dongbumlee/ontosmith.git
cd ontosmith
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov ruff   # test + lint tooling
```

Python ≥ 3.11. Optional surfaces install as extras:
`pip install -e ".[neo4j]"` (graph store), `".[nlp]"`
(spaCy + GLiNER extraction), `".[viz]"` (charts and
graph images). The core test suite runs on the base
install alone.

The `nlp` extra installs the spaCy and GLiNER
libraries. The GLiNER model downloads itself on first
load; the spaCy English model is a separate download:

```bash
python -m spacy download en_core_web_sm
```

Without the model, a batch run logs the miss and
continues without the structural pass — it does not
fail — so download the model before drawing conclusions
from a local run.

## Quality gates (all must pass)

```bash
python -m compileall -q pipeline harness tests packs api
ruff check .
python -m pytest tests/ -q --cov=pipeline --cov=harness --cov=api --cov-fail-under=45
```

- **Tests.** The full suite, green. New behavior lands
  with tests; a bug fix lands with the failing test that
  reproduced it.
- **Lint.** `ruff check .` clean (config in `ruff.toml`).
- **Coverage floor.** Coverage of `pipeline` + `harness` + `api`
  is a ratchet: the CI floor (45%) is the minimum, the
  measured baseline is recorded in the CI workflow, and a
  change may hold or raise it — never lower it.
- **Equivalence before/after.** Any refactor, and any
  change to pipeline behavior, ships with evidence that
  the same input produces the same output before and
  after: a replay, a golden-output test, or a recorded
  comparison. "It should be equivalent" is not evidence.

CI (`.github/workflows/ci.yml`) runs exactly these gates
on every push and pull request.

## Pack contributions

Packs are data with a measurement record, not prose:

- **A definition lives in exactly one place.** A pack's
  ontology, framing, and defaults are defined in its
  payload files; documentation and code reference them,
  never restate them. Two copies of a definition is a
  defect.
- **Content change = version bump.** Any change to a
  pack's payload bumps its manifest version in the same
  commit. The manifest hash is the identity a build ran
  under.
- **Conformance is measured, not asserted.** Claiming a
  pack supports a domain or a change improves it requires
  a registered measurement: a `conformance.json` entry
  naming the gate, the instrument, the population, and
  the delta. Negative results are recorded the same way
  — the conformance files in this repo carry measured
  rejections next to the adoptions, on purpose.

## Disciplines that are not negotiable

- **Evidence is shipped, claims are pointers.**
  Quality claims live in the whitepaper with their scope
  and method; the repository ships the data and the
  replay tooling that let a reader check them
  (`benchmarks/`). Do not add score claims to front
  surfaces (README, docs headers) — one pointer line to
  the record, nothing more.
- **Cost claims are aggregates.** A build's cost may
  be stated as the aggregate the run actually observed.
  Per-cell or per-stage spend breakdowns are internal
  provenance and do not ship — with one sanctioned
  exception: the whitepaper §5.6 reference-run table,
  whose per-stage figures ship as part of that run's
  aggregate record.
- **Fail closed.** Resolution, validation, and parsing
  refuse on ambiguity rather than guess: a pack that does
  not resolve is an error, not a fallback; an unapproved
  schema emits nothing. Keep it that way.
- **No credentials, ever.** API keys arrive via
  environment or the connection layer at run time; no
  key, token, or personal data lands in code, fixtures,
  or docs. Corpus text ships only as the project's own
  example corpora and the adjudication appendix's
  evidence excerpts, under their upstream terms;
  benchmark datasets themselves are fetch-only.

## Pull requests

Use the PR template (Summary / Why / What changed / How
verified / Checklist / Notes). The "How verified" section
is the heart of it: name the commands you ran and the
outputs you saw. A PR that changes behavior without
equivalence evidence will be asked for it.

## Reporting issues

Bug reports and hands-on feedback have issue templates.
For a measurement dispute, the most useful report is a
replay: run `examples/demo/replay_scores.py` (or your own
recomputation from `benchmarks/appendix/`) and attach
where your numbers leave the record's.
