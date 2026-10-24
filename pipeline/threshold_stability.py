"""Threshold-tuning validation: is the tuned t* stable and trustworthy?

Two checks, both offline (uses stored p_true + labels):
  1. bootstrap the tuning procedure on dev -> distribution of t*
     (stable selection = tight distribution)
  2. F1(t) curve on held-out -> flatness around t*
     (robust optimum = flat region, not a knife's edge)

Usage: python -m pipeline.threshold_stability [--result results/scale1k_jev-1.13_noul_v2.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_hardened import load, prf, split_by_doc, tune_threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default="results/scale1k_jev-1.13_noul_v2.json")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load(args.result)
    dev, held = split_by_doc(recs, seed=7)
    dev = list(dev)
    rng = np.random.default_rng(args.seed)

    ts = []
    for _ in range(args.iters):
        idx = rng.integers(0, len(dev), len(dev))
        ts.append(tune_threshold([dev[i] for i in idx]))
    ts = np.array(ts)

    out = {
        "t_star_mean": round(float(ts.mean()), 3),
        "t_star_median": round(float(np.median(ts)), 3),
        "t_star_sd": round(float(ts.std()), 3),
        "t_star_p5": round(float(np.percentile(ts, 5)), 2),
        "t_star_p95": round(float(np.percentile(ts, 95)), 2),
        "f1_curve": [],
    }
    for t in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
        m = prf(held, t)
        out["f1_curve"].append({"t": t, "held_f1": round(m["f1"], 3)})

    name = Path(args.result).stem
    Path(f"results/threshold_stability_{name}.json").write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
