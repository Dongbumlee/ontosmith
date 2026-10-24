"""The scoring instrument.

Extracted from a shared scorer and the flagship build's
aggregate, generalized to caller-supplied records: the caller
brings predicted records, gold records, and (for corrections)
verdicts; this module brings the arithmetic — and ONLY the
arithmetic. The estimator functions carry the method
verbatim in behavior (the estimator code is the method).

Record shapes (both are plain dicts):

- Predicted: ``{doc_id, subject, relation, object}``
  (``doc_index`` accepted in place of ``doc_id``).
- Gold: ``{doc_id, head_names, tail_names, relation}`` where
  heads/tails may be name lists, or the predicted shape with
  scalar ``subject``/``object`` (treated as one-name sets).

Scoring is exact-match over normalized keys, with an optional
:class:`NormalizationTable` (TN-1/TN-2): TN-1 sorts the
endpoints of declared-symmetric relations; TN-2 maps declared
inverse relations to their primary with endpoints swapped.
The table is DATA the caller supplies (a domain pack's
declarations), never code constants here — the lab scorer
read its table from a pack file; the product core takes the
same table as an input.

Corrections:

- corrected precision = (raw matches + judged-supported
  unmatched) / predicted — the unmatched pool's support rate
  comes from blind adjudication of a stratified sample, so
  the correction is an ESTIMATE with the sampling visible.
- ceiling-adjusted recall = matched gold / (gold x gold
  support rate) — gold itself is sampled for pair support,
  so the ceiling under 1.0 is measured, not assumed.
- :func:`stratified_estimate` combines per-stratum samples
  into a population estimate with a finite-population
  correction and a Wilson-style 95% interval; labels outside
  the decisive set are excluded from denominators and
  reported as counts (unknowns never silently vanish).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Z = 1.96  # 95% normal quantile, as in the lab instruments


def norm_text(s) -> str:
    """Casefold + whitespace-collapse normalization.

    Verbatim from the lab instrument (``" ".join(str(s)
    .casefold().split())``) — including its consequence that
    a missing value normalizes to the string "none": the
    scorer is the method, so its edge behavior is
    inherited deliberately, not silently repaired. Callers
    validate record completeness upstream.
    """
    return " ".join(str(s).casefold().split())


def wilson(k: int, n: int) -> tuple[float | None, float | None]:
    """Wilson 95% interval for a proportion k/n (lab formula)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def rate(k: int, n: int) -> dict:
    """A counted rate with its Wilson interval: {k, n, rate, ci}."""
    lo, hi = wilson(k, n)
    return {"k": k, "n": n,
            "rate": round(k / n, 4) if n else None,
            "wilson95": [lo, hi]}


def wilson_from_var(p: float, var: float) -> tuple[float, float]:
    """Wilson-style 95% interval for a stratified estimate,
    via the effective sample size n_eff = p(1-p)/Var."""
    if var <= 0 or p <= 0 or p >= 1:
        return (round(p, 4), round(p, 4))
    n_eff = p * (1 - p) / var
    d = 1 + Z * Z / n_eff
    c = (p + Z * Z / (2 * n_eff)) / d
    h = Z * math.sqrt(p * (1 - p) / n_eff + Z * Z / (4 * n_eff * n_eff)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


@dataclass(frozen=True)
class NormalizationTable:
    """TN-1/TN-2 relation canonicalization, as caller data.

    ``symmetric``: relation names whose endpoints sort
    (TN-1). ``inverses``: inverse relation -> primary
    relation; an inverse record canonicalizes to the primary
    with endpoints swapped (TN-2). An empty table (the
    default) canonicalizes nothing — plain exact match.
    """

    symmetric: frozenset = frozenset()
    inverses: dict = field(default_factory=dict)

    def canonical_key(self, doc_key, subject, relation, obj) -> tuple:
        """The canonical match key for one (subject, relation,
        object) triple in one document.

        Mirrors the lab instrument exactly: endpoints are
        normed, the relation label is matched VERBATIM (an
        inventory's labels are exact strings; case variants
        are distinct relations), TN-1 sorts the endpoints of
        a symmetric relation, else TN-2 maps an inverse to its
        primary with endpoints swapped.
        """
        s, o = norm_text(subject), norm_text(obj)
        r = str(relation or "")
        if r in self.symmetric:
            s, o = sorted((s, o))
            return (doc_key, s, r, o)
        primary = self.inverses.get(r)
        if primary is not None and r != primary:
            return (doc_key, o, primary, s)
        return (doc_key, s, r, o)


def _doc_key(rec: dict):
    """The record's document identity: doc_id, else doc_index."""
    return rec.get("doc_id", rec.get("doc_index"))


def record_key(rec: dict, table: NormalizationTable) -> tuple:
    """Canonical key of one predicted-shaped record."""
    return table.canonical_key(
        _doc_key(rec), rec.get("subject"), rec.get("relation"),
        rec.get("object"))


def gold_keys(rec: dict, table: NormalizationTable) -> set[tuple]:
    """All canonical keys a gold record can match: the cross
    product of its head and tail name sets (a gold triple is
    matched when ANY declared name pair matches)."""
    heads = rec.get("head_names") or [rec.get("subject")]
    tails = rec.get("tail_names") or [rec.get("object")]
    return {table.canonical_key(_doc_key(rec), h, rec.get("relation"), t)
            for h in heads for t in tails}


@dataclass
class MatchResult:
    """The partition of predicted x gold under exact match."""

    predicted_total: int
    gold_total: int
    matched_predicted: int
    matched_gold: int
    unmatched_predicted: list[dict]
    unmatched_gold: list[dict]
    unmatched_gold_indices: list[int] = field(default_factory=list)

    def raw_scores(self) -> dict:
        """Raw precision / recall / F1 with Wilson intervals."""
        return {
            "precision": rate(self.matched_predicted,
                              self.predicted_total),
            "recall": rate(self.matched_gold, self.gold_total),
            "matched_predicted": self.matched_predicted,
            "matched_gold": self.matched_gold,
            "predicted_total": self.predicted_total,
            "gold_total": self.gold_total,
        }


def match_records(predicted: list[dict], gold: list[dict],
                  table: NormalizationTable | None = None) -> MatchResult:
    """Partition predicted and gold records by exact match.

    Distinctness follows the lab scorer: predicted records
    count per canonical key (a duplicate key counts once);
    a gold record matches when any of its name-pair keys was
    predicted.
    """
    table = table or NormalizationTable()
    pred_by_key: dict[tuple, dict] = {}
    for rec in predicted:
        pred_by_key.setdefault(record_key(rec, table), rec)
    pred_key_set = set(pred_by_key)
    matched_pred_keys: set[tuple] = set()
    matched_gold_idx: set[int] = set()
    for i, grec in enumerate(gold):
        hit = gold_keys(grec, table) & pred_key_set
        if hit:
            matched_gold_idx.add(i)
            matched_pred_keys |= hit
    return MatchResult(
        predicted_total=len(pred_by_key),
        gold_total=len(gold),
        matched_predicted=len(matched_pred_keys),
        matched_gold=len(matched_gold_idx),
        unmatched_predicted=[rec for key, rec in pred_by_key.items()
                             if key not in matched_pred_keys],
        unmatched_gold=[rec for i, rec in enumerate(gold)
                        if i not in matched_gold_idx],
        unmatched_gold_indices=[i for i in range(len(gold))
                                if i not in matched_gold_idx])


def corrected_precision(matched: int, predicted_total: int,
                        judged_supported: float) -> float | None:
    """Audit-corrected precision (lab formula, verbatim):
    (matched + sample-estimated supported unmatched) /
    predicted. ``judged_supported`` is the estimated COUNT of
    supported records in the unmatched pool, not a rate."""
    if not predicted_total:
        return None
    return round((matched + judged_supported) / predicted_total, 4)


def ceiling_adjusted_recall(matched_gold: int, gold_total: int,
                            support_rate: float | None) -> float | None:
    """Recall against the measured gold ceiling (lab formula,
    verbatim): matched / (gold x support rate)."""
    if not gold_total or not support_rate:
        return None
    ceiling = gold_total * support_rate
    return round(matched_gold / ceiling, 4) if ceiling else None


def stratified_estimate(groups: list[dict],
                        positive: str = "supported",
                        unlabeled: str = "unlabeled") -> dict:
    """Stratified population estimate with CI.

    Each group: ``{name, population, sampled, labels}`` where
    ``labels`` is the multiset of verdict labels for the
    sampled items. The decisive support rate combines the
    strata weighted by population, with finite-population
    corrected variance; labels equal to ``unlabeled`` (or
    absent/None) are excluded from the decisive denominator
    and reported as counts; any other non-positive label
    counts as not-supported.
    """
    est_num = 0.0
    var = 0.0
    n_pop = n_samp = n_dec = n_unlab = 0
    per_group = {}
    for g in groups:
        labels = g.get("labels", [])
        counts: dict[str, int] = {}
        for lab in labels:
            key = lab if lab is not None else unlabeled
            counts[key] = counts.get(key, 0) + 1
        k = counts.get(positive, 0)
        n_un = counts.get(unlabeled, 0)
        n_g = len(labels) - n_un
        pop = float(g["population"])
        p_g = k / n_g if n_g else 0.0
        est_num += pop * p_g
        if n_g > 1 and pop > 0:
            var += (pop ** 2) * (p_g * (1 - p_g) / n_g) \
                * max(0.0, 1.0 - n_g / pop)
        per_group[g["name"]] = {"population": int(pop),
                                 "sampled": len(labels),
                                 "decisive": n_g, "supported": k,
                                 "rate": round(p_g, 4) if n_g else None}
        n_pop += int(pop)
        n_samp += len(labels)
        n_dec += n_g
        n_unlab += n_un
    p_hat = est_num / n_pop if n_pop else 0.0
    var_p = var / (n_pop ** 2) if n_pop else 0.0
    lo, hi = wilson_from_var(p_hat, var_p)
    return {
        "estimate": round(p_hat, 4),
        "wilson95": [lo, hi],
        "population": n_pop,
        "sampled": n_samp,
        "decisive": n_dec,
        "unlabeled": n_unlab,
        "estimated_supported_count": round(est_num, 2),
        "groups": per_group,
    }
