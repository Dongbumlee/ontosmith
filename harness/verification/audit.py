"""Gold-audit sampling and verdict resolution (plan §2.2).

Extracted from the lab's gold-audit set builder and the S3
verdict handling, generalized to caller records (the shapes
in :mod:`harness.verification.scoring`). The audit's job is
to measure the GOLD as well as the system: blind panels judge
samples of what the system emitted and what gold contains,
so scoring can correct for gold incompleteness instead of
pretending the reference is perfect.

The three judgment sets (the lab A/B/C design):

- **Set A** — unmatched predicted records: is the system
  right where gold is silent? (gold-incompleteness signal)
- **Set B** — sampled gold records: do gold's own pairs
  exist in the text, and is the relation type right?
  (gold-support and gold-type signals)
- **Set C** — matched predicted records: an honesty control;
  if the panel cannot confirm known matches, the panel —
  not the system — is the problem.

Sampling is seeded and stratified by document, deterministic
for a given (records, sizes, seed): re-running an audit-set
build reproduces the identical sets. Verdict resolution
follows the lab escalation rule: a later stage's decisive
verdict supersedes an earlier one; unknown/undecided
verdicts never supersede and never count as decisive.
"""
from __future__ import annotations

import random
import re

from .scoring import NormalizationTable, gold_keys, match_records, norm_text, record_key

#: Verdict values that carry no decision. Anything else a
#: panel returns is decisive (its exact vocabulary — e.g.
#: "supported" / "not_supported" — is the caller's contract).
UNKNOWN_VERDICTS = frozenset({None, "", "unknown", "unclear",
                              "undecided", "unlabeled"})


def stratified_sample(records: list[dict], n: int, seed: int,
                      key: str = "doc_id") -> list[dict]:
    """Draw ``n`` records spread across the ``key`` strata.

    Deterministic: groups are visited in sorted stratum
    order, each shuffled by a per-stratum RNG seeded from
    (seed, stratum rank); records are taken round-robin so
    the sample covers documents evenly before it deepens.
    When n >= len(records) the whole pool returns (sorted
    stratum order), matching the lab rule that a capped pool
    is judged whole.
    """
    groups: dict[object, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec.get(key, rec.get("doc_index")), []).append(rec)
    strata = sorted(groups, key=lambda s: str(s))
    pools = []
    for rank, stratum in enumerate(strata):
        pool = list(groups[stratum])
        random.Random(seed * 100003 + rank).shuffle(pool)
        pools.append(pool)
    if n >= len(records):
        return [rec for pool in pools for rec in pool]
    out: list[dict] = []
    depth = 0
    while len(out) < n:
        progressed = False
        for pool in pools:
            if depth < len(pool) and len(out) < n:
                out.append(pool[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return out


def build_audit_sets(predicted: list[dict], gold: list[dict],
                     sizes: dict | None = None, seed: int = 0,
                     table: NormalizationTable | None = None) -> dict:
    """Build the A/B/C judgment sets from predicted + gold.

    ``sizes`` maps "A"/"B"/"C" to sample sizes (absent = the
    whole pool). Item ids are per-set (A-001, B-001, C-001)
    and each item carries its source record under ``record``
    and its provenance under ``set``/``pool_size``, so the
    judgment frame is reproducible from the report alone.
    """
    sizes = sizes or {}
    table = table or NormalizationTable()
    m = match_records(predicted, gold, table)
    gold_key_set: set = set()
    for grec in gold:
        gold_key_set |= gold_keys(grec, table)
    matched_pred = [r for r in predicted
                    if record_key(r, table) in gold_key_set]

    def items(tag: str, pool: list[dict], n) -> list[dict]:
        chosen = pool if n is None else stratified_sample(pool, n, seed)
        return [{"item_id": f"{tag}-{i + 1:03d}", "set": tag,
                 "pool_size": len(pool), "record": rec}
                for i, rec in enumerate(chosen)]

    return {
        "A": items("A", m.unmatched_predicted, sizes.get("A")),
        "B": items("B", gold, sizes.get("B")),
        "C": items("C", matched_pred, sizes.get("C")),
        "frame": {
            "predicted_total": m.predicted_total,
            "gold_total": m.gold_total,
            "matched_predicted": m.matched_predicted,
            "matched_gold": m.matched_gold,
            "seed": seed,
        },
    }


def final_verdicts(*stages: list[dict], field: str = "verdict",
                   id_field: str = "item_id") -> dict:
    """Resolve final verdicts across escalation stages.

    Stages are given earliest-first; a later stage's DECISIVE
    verdict replaces an earlier one for the same item; unknown
    verdicts (see :data:`UNKNOWN_VERDICTS`) never replace and
    never enter the result. This is the lab rule that the
    escalation seat's decisive reading is the final verdict.
    """
    out: dict[str, str] = {}
    for stage in stages:
        for row in stage:
            verdict = row.get(field)
            if verdict in UNKNOWN_VERDICTS:
                continue
            out[str(row[id_field])] = verdict
    return out


def decisive_counts(verdicts: dict, positive: str = "supported") -> dict:
    """{decisive, positive} counts over resolved verdicts."""
    vals = list(verdicts.values())
    return {"decisive": len(vals),
            "positive": sum(1 for v in vals if v == positive)}


def clean_quote(q: str) -> str:
    """Strip one layer of wrapping quotes (lab clean_quote)."""
    q = (q or "").strip()
    if len(q) >= 2 and q[0] == q[-1] and q[0] in "\"'“”":
        q = q[1:-1].strip()
    return q


def quote_verified(evidence: str, context: str) -> bool:
    """True iff the (cleaned) evidence quote appears verbatim
    in the context, whitespace-normalized (lab rule)."""
    q = clean_quote(evidence)
    if not q:
        return False
    ws = re.compile(r"\s+")
    return ws.sub(" ", q) in ws.sub(" ", context or "")


__all__ = ["UNKNOWN_VERDICTS", "build_audit_sets", "clean_quote",
           "decisive_counts", "final_verdicts", "norm_text",
           "quote_verified", "stratified_sample"]
