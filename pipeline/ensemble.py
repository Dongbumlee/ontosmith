"""Offline ensemble analysis: join noul + choice runs on identical candidates.

No API calls. Evaluates combination rules:
  - noul_only / choice_only (baselines)
  - and_rule: accept iff both gates accept
  - or_rule: accept iff either gate accepts
  - routed: both agree -> auto decision; disagree -> human review.
    Reports review rate and accuracy/precision on the auto-decided subset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def key(r: dict) -> tuple:
    return (r["doc_id"], r["subject"], r["relation"], r["object"])


def load(name: str) -> dict[tuple, dict]:
    d = json.loads((RESULTS_DIR / name).read_text())
    recs = {key(r): r for r in d["records"]}
    return recs


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    n = tp + fp + fn + tn
    return {
        "n": n,
        "accuracy": (tp + tn) / n if n else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noul", default="v0_hard_results.json")
    ap.add_argument("--choice", default="v0_hard_choice_results.json")
    args = ap.parse_args()

    noul = load(args.noul)
    choice = load(args.choice)
    common = [k for k in noul if k in choice]
    print(f"joined candidates: {len(common)} (noul={len(noul)}, choice={len(choice)})")

    labels = {k: noul[k]["label"] for k in common}
    n_pred = {k: 1 if noul[k]["p_true"] >= 0.5 else 0 for k in common}
    c_pred = {k: 1 if choice[k]["p_true"] >= 0.5 else 0 for k in common}

    def counts(pred: dict) -> tuple[int, int, int, int]:
        tp = sum(1 for k in common if pred[k] == 1 and labels[k] == 1)
        fp = sum(1 for k in common if pred[k] == 1 and labels[k] == 0)
        fn = sum(1 for k in common if pred[k] == 0 and labels[k] == 1)
        tn = sum(1 for k in common if pred[k] == 0 and labels[k] == 0)
        return tp, fp, fn, tn

    rules = {
        "noul_only": n_pred,
        "choice_only": c_pred,
        "and_rule": {k: n_pred[k] & c_pred[k] for k in common},
        "or_rule": {k: n_pred[k] | c_pred[k] for k in common},
    }
    for name, pred in rules.items():
        m = prf(*counts(pred))
        print(f"{name:12s} acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
              f"rec={m['recall']:.3f}")

    # 3-way routing: agree -> auto, disagree -> human review
    auto = [k for k in common if n_pred[k] == c_pred[k]]
    review = [k for k in common if n_pred[k] != c_pred[k]]
    auto_pred = {k: n_pred[k] for k in auto}
    tp = sum(1 for k in auto if auto_pred[k] == 1 and labels[k] == 1)
    fp = sum(1 for k in auto if auto_pred[k] == 1 and labels[k] == 0)
    fn = sum(1 for k in auto if auto_pred[k] == 0 and labels[k] == 1)
    tn = sum(1 for k in auto if auto_pred[k] == 0 and labels[k] == 0)
    m = prf(tp, fp, fn, tn)
    # disagreement direction: which gate said accept?
    noul_only_accept = sum(1 for k in review if n_pred[k] == 1)
    print(f"\nrouted: auto-decided {len(auto)}/{len(common)} "
          f"({len(auto)/len(common):.1%}), human review {len(review)} "
          f"({len(review)/len(common):.1%})")
    print(f"  on auto-decided: acc={m['accuracy']:.3f} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f}")
    print(f"  disagreements where noul accepted / choice rejected: {noul_only_accept}, "
          f"vice versa: {len(review) - noul_only_accept}")
    # what fraction of review items are actually positive?
    rev_pos = sum(labels[k] for k in review)
    print(f"  review set: {rev_pos}/{len(review)} are true triples "
          f"({rev_pos/len(review):.1%} prevalence vs "
          f"{sum(labels.values())/len(common):.1%} overall)")


if __name__ == "__main__":
    main()
