"""Entity-resolution candidate construction from DocRED vertexSet.

Positives: two mentions of the SAME entity with different surface names.
Hard negatives:
  diff_entity  - mentions of two DIFFERENT entities in the same doc
  shared_token - different entities whose mentions share a token
                 (e.g. 'Skai TV' vs 'Skai Group')
"""
from __future__ import annotations

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dev():
    return json.loads((DATA_DIR / "dev.json").read_text())


def sent_text(doc, sid):
    return " ".join(doc["sents"][sid])


def build_er_candidates(n_docs=400, n_total=1000, seed=123):
    rng = random.Random(seed)
    docs = json.loads((DATA_DIR / "dev.json").read_text())
    docs = rng.sample(docs, n_docs)
    pos, negs = [], {"diff_entity": [], "shared_token": []}
    for di, doc in enumerate(docs):
        vs = doc["vertexSet"]
        # positives: same entity, distinct surface names
        for ent in vs:
            names = {}
            for m in ent:
                names.setdefault(m["name"], m)
            if len(names) >= 2:
                ms = list(names.values())
                a, b = rng.sample(ms, 2)
                ctx = sent_text(doc, a["sent_id"]) + " [SEP] " + sent_text(doc, b["sent_id"])
                pos.append({
                    "doc_id": f"er{di}", "text": doc_text(doc), "context": ctx,
                    "mention_a": a["name"], "mention_b": b["name"],
                    "label": 1, "neg_kind": None,
                })
        # negatives
        flat = [(ei, m) for ei, ent in enumerate(vs) for m in ent]
        for _ in range(40):  # attempts per doc
            (ei, a), (ej, b) = rng.sample(flat, 2)
            if ei == ej or a["name"] == b["name"]:
                continue
            ta, tb = set(a["name"].lower().split()), set(b["name"].lower().split())
            kind = "shared_token" if ta & tb else "diff_entity"
            ctx = sent_text(doc, a["sent_id"]) + " [SEP] " + sent_text(doc, b["sent_id"])
            negs[kind].append({
                "doc_id": f"er{di}", "text": doc_text(doc), "context": ctx,
                "mention_a": a["name"], "mention_b": b["name"],
                "label": 0, "neg_kind": kind,
            })
    # balance: 500 pos / 500 neg (250 per kind)
    rng.shuffle(pos)
    for k in negs:
        rng.shuffle(negs[k])
    cands = pos[:500] + negs["diff_entity"][:250] + negs["shared_token"][:250]
    rng.shuffle(cands)
    return cands


def doc_text(doc):
    return " ".join(" ".join(s) for s in doc["sents"])


if __name__ == "__main__":
    cands = build_er_candidates()
    out = Path(__file__).resolve().parent.parent / "results" / "candidates_er1000.json"
    out.write_text(json.dumps(cands, ensure_ascii=False))
    from collections import Counter
    print(len(cands), Counter((c["label"], c["neg_kind"]) for c in cands))
    print("wrote", out)
