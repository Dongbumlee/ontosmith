"""Hardened analysis for publication.

- Doc-level dev/held-out split (no leakage: same doc never in both)
- Thresholds tuned on dev, metrics reported on held-out
- Cluster bootstrap (over docs) 95% CIs for F1
- McNemar's test for paired model comparisons on identical candidates

Usage: python -m pilot.analyze_hardened results/matrix_<model>_<gate>.json [...]
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())["records"]


def split_by_doc(records: list[dict], seed: int = 7, frac: float = 0.5):
    docs = sorted({r["doc_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(docs)
    cut = int(len(docs) * frac)
    dev_docs = set(docs[:cut])
    dev = [r for r in records if r["doc_id"] in dev_docs]
    held = [r for r in records if r["doc_id"] not in dev_docs]
    return dev, held


def prf(records: list[dict], t: float) -> dict:
    tp = fp = fn = tn = 0
    for r in records:
        pred = 1 if r["p_true"] >= t else 0
        lab = r["label"]
        if pred == 1 and lab == 1:
            tp += 1
        elif pred == 1 and lab == 0:
            fp += 1
        elif pred == 0 and lab == 1:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": (tp + tn) / n, "prec": prec, "rec": rec, "f1": f1, "n": n}


def tune_threshold(dev: list[dict]) -> float:
    best_t, best_f1 = 0.5, -1.0
    for i in range(1, 20):
        t = i / 20
        f1 = prf(dev, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def bootstrap_f1(records: list[dict], t: float, iters: int = 1000,
                 seed: int = 11) -> tuple[float, float]:
    """Cluster bootstrap over docs; returns 95% CI for F1."""
    rng = random.Random(seed)
    by_doc: dict[str, list[dict]] = {}
    for r in records:
        by_doc.setdefault(r["doc_id"], []).append(r)
    docs = list(by_doc.keys())
    f1s = []
    for _ in range(iters):
        sample = []
        for _ in docs:
            sample.extend(by_doc[rng.choice(docs)])
        f1s.append(prf(sample, t)["f1"])
    f1s.sort()
    return f1s[int(0.025 * iters)], f1s[int(0.975 * iters)]


def mcnemar(a: list[dict], b: list[dict], t_a: float, t_b: float) -> dict:
    """McNemar on paired accept/reject correctness (same candidates)."""
    ka = {(r["doc_id"], r["subject"], r["relation"], r["object"]): r for r in a}
    kb = {(r["doc_id"], r["subject"], r["relation"], r["object"]): r for r in b}
    common = [k for k in ka if k in kb]
    b01 = b10 = 0  # a wrong b right / a right b wrong
    for k in common:
        ca = (ka[k]["p_true"] >= t_a) == bool(ka[k]["label"])
        cb = (kb[k]["p_true"] >= t_b) == bool(kb[k]["label"])
        if not ca and cb:
            b01 += 1
        elif ca and not cb:
            b10 += 1
    n = b01 + b10
    if n == 0:
        return {"n_common": len(common), "b01": 0, "b10": 0, "p_value": 1.0}
    # chi-square with continuity correction
    chi2 = (abs(b01 - b10) - 1) ** 2 / n
    # p-value from chi-square(1): use survival approximation
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"n_common": len(common), "b01": b01, "b10": b10,
            "chi2": round(chi2, 3), "p_value": round(p, 4)}


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print("usage: python -m pilot.analyze_hardened <result json> ...")
        return
    cfgs = {}
    for p in paths:
        name = Path(p).stem.replace("matrix_", "")
        recs = load(p)
        dev, held = split_by_doc(recs)
        t = tune_threshold(dev)
        m = prf(held, t)
        lo, hi = bootstrap_f1(held, t)
        cfgs[name] = {"recs": recs, "t": t, "held": m, "ci": (lo, hi),
                      "n_dev": len(dev), "n_held": len(held)}
        print(f"{name:40s} t*={t:.2f} heldout: "
              f"acc={m['acc']:.3f} prec={m['prec']:.3f} rec={m['rec']:.3f} "
              f"F1={m['f1']:.3f} [{lo:.3f},{hi:.3f}] "
              f"(dev={len(dev)}, held={len(held)})")

    # McNemar: jev vs msft per gate (paired on identical candidates)
    for gate in ["noul", "noul_v2", "choice", "score"]:
        a = f"jev-1.13_{gate}"
        b = f"microsoft-decision-1_{gate}"
        if a in cfgs and b in cfgs:
            r = mcnemar(cfgs[a]["recs"], cfgs[b]["recs"],
                        cfgs[a]["t"], cfgs[b]["t"])
            sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "not significant"
            print(f"McNemar jev vs msft [{gate}]: {r} -> {sig}")


if __name__ == "__main__":
    main()
