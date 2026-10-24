"""Hybrid retrieval eval: vector-only vs vector + graph traversal.

QA pairs from DocRED gold triples (1-hop and 2-hop). Compares:
  - vector: top-k docs by embedding similarity; hit if answer in doc text
  - hybrid: top-3 docs -> entities -> Neo4j 1..2 hop traversal;
            hit if answer entity in traversed subgraph

Usage: python -m pipeline.retrieval_eval [--n1 100] [--n2 100] [--out results/retrieval.json]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path


def norm(x: str) -> str:
    return " ".join(x.strip().lower().split())


def build_qa(cands, n1: int, n2: int, seed: int):
    rng = random.Random(seed)
    pos = [c for c in cands if c.get("label") == 1]
    by_doc: dict[str, list] = defaultdict(list)
    for c in pos:
        by_doc[c["doc_id"]].append(c)

    one, two = [], []
    # 1-hop
    pool = [c for c in pos if c.get("text")]
    rng.shuffle(pool)
    for c in pool:
        if len(one) >= n1:
            break
        one.append({
            "q": f"What is the {c['relation']} of {c['subject']}?",
            "a": c["object"], "hops": 1, "doc_id": c["doc_id"],
        })
    # 2-hop chains: within-doc AND cross-doc (bridge entity in two docs)
    # index: norm(object name) -> list of triples where it is the subject
    by_subj: dict[str, list] = defaultdict(list)
    for c in pos:
        by_subj[norm(c["subject"])].append(c)
    chains = []
    for c1 in pos:
        for c2 in by_subj.get(norm(c1["object"]), []):
            if norm(c2["object"]) == norm(c1["subject"]):
                continue
            if c1["doc_id"] == c2["doc_id"] and (c1["subject"], c1["relation"], c1["object"]) == (c2["subject"], c2["relation"], c2["object"]):
                continue
            chains.append((c1, c2))
    rng.shuffle(chains)
    for c1, c2 in chains:
        if len(two) >= n2:
            break
        two.append({
            "q": (f"What is the {c2['relation']} of the "
                  f"{c1['relation']} of {c1['subject']}?"),
            "a": c2["object"], "hops": 2,
            "doc_id": c1["doc_id"], "bridge": c1["object"],
            "cross_doc": c1["doc_id"] != c2["doc_id"],
        })
    return one + two


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=100)
    ap.add_argument("--n2", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/retrieval.json")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    import numpy as np
    from fastembed import TextEmbedding

    cands = json.load(open("results/candidates_s1000.json"))
    qa = build_qa(cands, args.n1, args.n2, args.seed)
    print(f"QA: {sum(1 for q in qa if q['hops']==1)} 1-hop, "
          f"{sum(1 for q in qa if q['hops']==2)} 2-hop")

    # doc corpus
    docs: dict[str, str] = {}
    doc_ents: dict[str, set] = defaultdict(set)
    for c in cands:
        d = c["doc_id"]
        if c.get("text") and d not in docs:
            docs[d] = c["text"]
        doc_ents[d].add(c["subject"])
        doc_ents[d].add(c["object"])
    doc_ids = sorted(docs)
    print(f"corpus: {len(doc_ids)} docs")

    st = TextEmbedding(args.model)
    doc_emb = np.array(list(st.embed([docs[d] for d in doc_ids])))
    q_emb = np.array(list(st.embed([q["q"] for q in qa])))
    # L2-normalize for cosine
    doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)

    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        sys.exit("set NEO4J_PASSWORD")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pw))

    # entity name -> exists in graph (lowercased index)
    with driver.session() as s:
        names = [r["n"] for r in s.run("MATCH (e:Entity) RETURN e.name AS n")]
    gnames = {norm(n) for n in names}
    name_map = {norm(n): n for n in names}

    def traverse(entry_names: set[str]) -> set[str]:
        found: set[str] = set()
        with driver.session() as s:
            for en in entry_names:
                key = norm(en)
                if key not in name_map:
                    continue
                rows = s.run(
                    "MATCH (e:Entity {name: $n})-[*1..2]-(x:Entity) "
                    "RETURN DISTINCT x.name AS x",
                    n=name_map[key])
                for r in rows:
                    found.add(norm(r["x"]))
        return found

    sims = q_emb @ doc_emb.T  # cosine (normalized)
    res = []
    for i, q in enumerate(qa):
        top5 = np.argsort(-sims[i])[:5]
        top3 = top5[:3]
        ans = norm(q["a"])
        # vector-only: answer string in any top-5 doc
        v_hit = any(ans in norm(docs[doc_ids[j]]) for j in top5)
        # hybrid
        entry: set[str] = set()
        for j in top3:
            entry |= {e for e in doc_ents[doc_ids[j]] if norm(e) in gnames}
        sub = traverse(entry)
        h_hit = ans in sub
        res.append({"q": q["q"], "a": q["a"], "hops": q["hops"],
                    "cross_doc": q.get("cross_doc", False),
                    "vector_hit": bool(v_hit), "hybrid_hit": bool(h_hit),
                    "n_entry": len(entry), "n_sub": len(sub)})

    def recall(rs, key):
        return sum(1 for r in rs if r[key]) / len(rs) if rs else 0.0

    out = {
        "n_qa": len(qa),
        "vector_recall": recall(res, "vector_hit"),
        "hybrid_recall": recall(res, "hybrid_hit"),
        "by_hop": {},
        "records": res,
    }
    for h in (1, 2):
        rs = [r for r in res if r["hops"] == h]
        out["by_hop"][str(h)] = {
            "n": len(rs),
            "vector_recall": recall(rs, "vector_hit"),
            "hybrid_recall": recall(rs, "hybrid_hit"),
        }
    # cross-doc 2-hop breakdown
    xd = [r for r in res if r["hops"] == 2 and r.get("cross_doc")]
    wd = [r for r in res if r["hops"] == 2 and not r.get("cross_doc")]
    out["two_hop_within_doc"] = {"n": len(wd), "vector_recall": recall(wd, "vector_hit"),
                                 "hybrid_recall": recall(wd, "hybrid_hit")}
    out["two_hop_cross_doc"] = {"n": len(xd), "vector_recall": recall(xd, "vector_hit"),
                                "hybrid_recall": recall(xd, "hybrid_hit")}
    # McNemar on paired hits
    b = sum(1 for r in res if r["hybrid_hit"] and not r["vector_hit"])
    c = sum(1 for r in res if r["vector_hit"] and not r["hybrid_hit"])
    out["mcnemar"] = {"hybrid_only": b, "vector_only": c}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "records"}, indent=1))
    driver.close()


if __name__ == "__main__":
    main()
