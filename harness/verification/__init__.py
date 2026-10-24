"""Reusable verification cores.

Extracted from per-cell scripts into domain-neutral,
parameterized form:

- :mod:`harness.verification.config` — pre-registration as data
  structures: gate definitions, counted cost projections,
  stop-lines and spend ledgers, and the frontier-escalation
  policy (off by default; the judging seat never
  escalates).
- :mod:`harness.verification.scoring` — the scoring instrument:
  exact-match scoring with an optional normalization table,
  audit-corrected precision, ceiling-adjusted recall, and the
  stratified estimator with confidence intervals.
- :mod:`harness.verification.audit` — gold-audit sampling:
  stratified frames, the A/B/C judgment sets, decisive-verdict
  resolution across escalation stages, and quote verification.
- :mod:`harness.verification.cell_runner` — the pre-registered
  adoption-cell runner: projection gate, per-item ledger with
  stop-line, checkpoint/resume, and gate evaluation.
- :mod:`harness.verification.report` — the one-command
  evaluation report: raw scores + corrected estimates +
  failure anatomy, as JSON and Markdown.

Nothing here names a corpus, a pack, or a model: corpora, gold,
and bindings arrive as inputs, and prices are caller-supplied
data (the connection layer's business), never constants.
"""
from __future__ import annotations

from . import audit, cell_runner, config, report, scoring

__all__ = ["audit", "cell_runner", "config", "report", "scoring"]
