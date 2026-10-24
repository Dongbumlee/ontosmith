"""BioRED candidate builder: same schema as DocRED candidates.

{doc_id, text, subject, relation, object, label}
Positives from gold relations; hard negatives via entity_swap,
relation_confusion, direction_flip (same recipe as hard_negatives.py).

Usage: python -m pipeline.biored_candidates --out results/candidates_biored1000.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "biored" / "BioRED"

# semantically confusable BioRED relation types
CONFUSION = {
    "Positive_Correlation": ["Negative_Correlation", "Association"],
    "Negative_Correlation": ["Positive_Correlation", "Association"],
    "Association": ["Positive_Correlation", "Negative_Correlation"],
    "Bind": ["Association"],
    "Cotreatment": ["Association", "Comparison"],
    "Comparison": ["Association", "Cotreatment"],
}
SYMMETRIC = {"Association", "Positive_Correlation", "Negative_Correlation",
             "Bind", "Cotreatment", "Comparison"}


def load_docs(split: str = "Train") -> list[dict]:
    d = json.loads((DATA / f"{split}.BioC.JSON").read_text())
    return d["documents"]


def doc_text(doc: dict) -> str:
    return " ".join(p["text"] for p in doc.get("passages", []))


def entity_mentions(doc: dict) -> dict[str, list[str]]:
    """identifier -> [mention texts]"""
    m: dict[str, list[str]] = {}
    for p in doc.get("passages", []):
        for a in p.get("annotations", []):
            ident = a["infons"].get("identifier", "")
            if ident:
                m.setdefault(ident, []).append(a["text"])
    return m


def build(n_docs: int = 200, n_total: int = 1000, seed: int = 123,
          split: str = "Train") -> list[dict]:
    rng = random.Random(seed)
    docs = rng.sample(load_docs(split), n_docs)
    cands: list[dict] = []
    per_doc = max(1, n_total // n_docs // 2)
    for doc in docs:
        text = doc_text(doc)
        mentions = entity_mentions(doc)
        gold = []
        for r in doc.get("relations", []):
            e1, e2 = r["infons"]["entity1"], r["infons"]["entity2"]
            t1 = mentions.get(e1, [e1])[0]
            t2 = mentions.get(e2, [e2])[0]
            rel = r["infons"]["type"]
            gold.append((t1, rel, t2))
        if not gold:
            continue
        # positives (sample)
        for t1, rel, t2 in rng.sample(gold, min(per_doc, len(gold))):
            cands.append({"doc_id": doc["id"], "text": text, "subject": t1,
                          "relation": rel, "object": t2, "label": 1,
                          "neg_kind": None})
        # hard negatives
        ids = list(mentions.keys())
        for t1, rel, t2 in rng.sample(gold, min(per_doc, len(gold))):
            kind = rng.choice(["entity_swap", "relation_confusion",
                               "direction_flip"])
            if kind == "entity_swap" and len(ids) >= 3:
                e_swap = rng.choice(ids)
                t_swap = mentions[e_swap][0]
                cands.append({"doc_id": doc["id"], "text": text,
                              "subject": t_swap, "relation": rel,
                              "object": t2, "label": 0,
                              "neg_kind": "entity_swap"})
            elif kind == "relation_confusion" and rel in CONFUSION:
                wrong = rng.choice(CONFUSION[rel])
                cands.append({"doc_id": doc["id"], "text": text,
                              "subject": t1, "relation": wrong,
                              "object": t2, "label": 0,
                              "neg_kind": "relation_confusion"})
            else:  # direction_flip
                if rel in SYMMETRIC:
                    # symmetric: flip is still true-ish; use confusion instead
                    wrong = rng.choice(CONFUSION.get(rel, ["Association"]))
                    cands.append({"doc_id": doc["id"], "text": text,
                                  "subject": t1, "relation": wrong,
                                  "object": t2, "label": 0,
                                  "neg_kind": "relation_confusion"})
                else:
                    cands.append({"doc_id": doc["id"], "text": text,
                                  "subject": t2, "relation": rel,
                                  "object": t1, "label": 0,
                                  "neg_kind": "direction_flip"})
    rng.shuffle(cands)
    return cands[:n_total]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/candidates_biored1000.json")
    ap.add_argument("--n-docs", type=int, default=200)
    ap.add_argument("--n-total", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--split", default="Train")
    args = ap.parse_args()
    cands = build(args.n_docs, args.n_total, args.seed, args.split)
    Path(args.out).write_text(json.dumps(cands, ensure_ascii=False))
    from collections import Counter
    print(f"Wrote {args.out}: {len(cands)} "
          f"(pos={sum(1 for c in cands if c['label']==1)})")
    print("neg kinds:", dict(Counter(c["neg_kind"] for c in cands
                                     if c["label"] == 0)))


if __name__ == "__main__":
    main()
