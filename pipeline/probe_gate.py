"""Probe gate — decide per document whether pair enumeration runs.

The pair census (pipeline/pair_proposer.enumerate_pairs) can be
large on roster-rich documents, and on many documents almost none
of its pairs state a relation. The probe gate buys the decision
with a bounded sample instead of a genre or pack label:

1. Draw a fixed-size random sample (the probe) from the document's
   census, deterministically seeded so a run and its measurement
   tooling cannot drift apart.
2. Judge the probe pairs through the ordinary pair-judge path.
3. Open the full census for that document only when the probe hit
   rate (share of probe pairs judged positive) reaches the floor.

This module is the pure decision logic only: sampling, hit-rate
arithmetic and the open/close decision. It makes no model calls —
the caller supplies the probe verdicts. The constants are global
policy constants (one probe size, one floor), not per-pack data:
the phase-3 probe-gate measurement (2026-11-09) found the pair
(probe 40, floor >= 5%) reproducing the full-census decision
across legal, fibo, bio-medical and open-domain corpora, while a
10% floor produced false opens and a strict-greater reading of
the 5% floor produced false closes on boundary documents.
"""
from __future__ import annotations

import random

PROBE_N = 40
PROBE_MIN_HIT_RATE = 0.05
PROBE_SEED = 20261109


def select_probe(pairs: list, probe_n: int = PROBE_N,
                 seed: int = PROBE_SEED) -> list:
    """The probe sample for one document's census.

    A census no larger than the probe is its own probe (the
    sample is the whole census, order preserved). Larger censuses
    are sampled without replacement under the fixed seed — the
    same draw every time the same census is probed, so gate
    decisions are reproducible and auditable after the fact."""
    if probe_n < 1:
        raise ValueError(f"probe_n must be >= 1, got {probe_n}")
    if len(pairs) <= probe_n:
        return list(pairs)
    return random.Random(seed).sample(pairs, probe_n)


def hit_rate(relations: list[str]) -> float:
    """Share of judged pairs that came back positive.

    `relations` is one judged relation name per probe pair, with
    "NONE" (any case) for a negative verdict — the pair judge's
    own vocabulary. An empty probe has hit rate 0.0."""
    if not relations:
        return 0.0
    hits = sum(1 for r in relations
               if r and str(r).strip().upper() != "NONE")
    return hits / len(relations)


def probe_decision(relations: list[str],
                   floor: float = PROBE_MIN_HIT_RATE) -> dict:
    """The gate decision for one document from its probe verdicts.

    Opens when the hit rate reaches the floor — the comparison is
    >= on purpose: a probe at exactly the floor (2/40 at the
    default) is an open, because boundary probes measured closer
    to the full-census opens than to the closes. An empty probe
    (empty census, or a census that produced nothing to judge)
    never opens: there is nothing for enumeration to add."""
    rate = hit_rate(relations)
    return {
        "open": bool(relations) and rate >= floor,
        "probe_n": len(relations),
        "hits": sum(1 for r in relations
                    if r and str(r).strip().upper() != "NONE"),
        "hit_rate": rate,
        "floor": floor,
    }
