"""Operating-point analysis for publication.

Beyond max-F1, reports production-relevant operating points, all tuned on
the dev split and evaluated on the held-out split (doc-level, no leakage):

  1. max-F1            — standard, comparable
  2. prec>=0.95        — max recall subject to precision >= 0.95
                         (graph contamination is costlier than misses)
  3. routed            — two thresholds (t_lo, t_hi): accept if p >= t_hi,
                         reject if p < t_lo, else human review.
                         Objective: min review rate s.t. auto precision >= 0.90

Usage: python -m pipeline.operating_points <result json> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

from .analyze_hardened import load, prf, split_by_doc


def sweep(records):
    """Return list of (t, prf) for t in 0.05..0.95."""
    return [(round(i / 100, 2), prf(records, i / 100)) for i in range(5, 96, 5)]


def op_max_f1(dev):
    return max(sweep(dev), key=lambda x: x[1]["f1"])[0]


def op_prec_constrained(dev, min_prec=0.95):
    cands = [(t, m) for t, m in sweep(dev) if m["prec"] >= min_prec]
    if not cands:
        # fall back: highest precision point
        return max(sweep(dev), key=lambda x: (x[1]["prec"], x[1]["rec"]))[0]
    return max(cands, key=lambda x: x[1]["rec"])[0]


def op_routed(dev, min_auto_prec=0.90):
    """Grid-search (t_lo, t_hi); minimize review rate s.t. auto precision >= target."""
    best = None
    for i in range(5, 96, 5):
        for j in range(i + 5, 101, 5):
            t_lo, t_hi = i / 100, j / 100
            auto = [r for r in dev if r["p_true"] >= t_hi or r["p_true"] < t_lo]
            review = [r for r in dev if t_lo <= r["p_true"] < t_hi]
            if not auto:
                continue
            # accept iff p >= t_hi (computed directly below)
            tp = sum(1 for r in auto if r["p_true"] >= t_hi and r["label"] == 1)
            fp = sum(1 for r in auto if r["p_true"] >= t_hi and r["label"] == 0)
            n_accept = tp + fp
            # require minimum support: vacuous precision (no accepts) is invalid
            if n_accept < max(10, int(0.05 * len(dev))):
                continue
            prec = tp / n_accept
            rr = len(review) / len(dev)
            if prec >= min_auto_prec and (best is None or rr < best[3]):
                best = (t_lo, t_hi, prec, rr)
    return best


def report(dev, held, t, t_lo=None, t_hi=None):
    if t_lo is None:
        m = prf(held, t)
        return {"t": t, **{k: round(v, 3) for k, v in m.items() if k != "n"}}
    auto = [r for r in held if r["p_true"] >= t_hi or r["p_true"] < t_lo]
    review = [r for r in held if t_lo <= r["p_true"] < t_hi]
    tp = sum(1 for r in auto if r["p_true"] >= t_hi and r["label"] == 1)
    fp = sum(1 for r in auto if r["p_true"] >= t_hi and r["label"] == 0)
    fn = sum(1 for r in auto if r["p_true"] < t_lo and r["label"] == 1)
    tn = sum(1 for r in auto if r["p_true"] < t_lo and r["label"] == 0)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "t_lo": t_lo, "t_hi": t_hi,
        "review_rate": round(len(review) / len(held), 3),
        "auto_acc": round((tp + tn) / n, 3) if n else 0.0,
        "auto_prec": round(prec, 3), "auto_rec": round(rec, 3),
    }


def main() -> None:
    for p in sys.argv[1:]:
        name = Path(p).stem
        recs = load(p)
        dev, held = split_by_doc(recs)
        t_f1 = op_max_f1(dev)
        t_p95 = op_prec_constrained(dev)
        routed = op_routed(dev)
        print(f"== {name} (dev={len(dev)}, held={len(held)})")
        print("  max-F1      :", report(dev, held, t_f1))
        print("  prec>=0.95  :", report(dev, held, t_p95))
        if routed:
            print("  routed      :", report(dev, held, None, routed[0], routed[1]))
        else:
            print("  routed      : no (t_lo,t_hi) reached auto-prec>=0.90 on dev")


if __name__ == "__main__":
    main()
