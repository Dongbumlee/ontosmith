"""Hybrid search eval: BM25 + dense (RRF) vs dense-only, same QA as rag_eval.

Motivation (2026-10-12): modern vector engines ship hybrid
search (dense + BM25) by default; a pure-dense baseline understates RAG.
Gold criterion (exact entity string in chunk) favors keyword matching,
so this tests how much of the graph's edge is an artifact of the baseline.

QA sets identical to rag_eval.py (seed 42). Dense = text-embedding-3-large
(cached tier2_index_large.pkl).
"""
import json
import pickle
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

random.seed(42)
BATCH = Path("batches/jfk-hybrid-v1")

# --- QA construction (identical to rag_eval.py) ---
tier1, tier2 = [], []
for l in open(BATCH / "routed.jsonl"):
    r = json.loads(l)
    t = (r["subject"], r["relation"], r["object"])
    (tier1 if r.get("tier") == 1 else tier2 if r.get("tier") == 2 else []).append(t)
tier1 = list(set(tier1)); tier2 = list(set(tier2))
t1_verbs = [t for t in tier1 if t[1] != "associated_with"]
t2_verbs = [t for t in tier2 if t[1] != "associated_with"]

qa1 = [(f"What did {s} {rel}?", [o]) for s, rel, o in random.sample(t1_verbs, 60)]
by_subj = defaultdict(list)
for s, rel, o in tier1:
    by_subj[s.lower()].append((s, rel, o))
paths = []
for s, rel, o in tier1:
    for s2, rel2, o2 in by_subj.get(o.lower(), []):
        if o2.lower() != s.lower():
            paths.append((s, rel, o, rel2, o2))
paths = random.sample(paths, min(40, len(paths)))
qa2 = [(f"What is connected to {s} through {mid}?", [mid, o2]) for s, r1_, mid, r2_, o2 in paths]
qa3 = [(f"What did {s} {rel}?", [o]) for s, rel, o in random.sample(t2_verbs, min(60, len(t2_verbs)))]
print(f"QA: single {len(qa1)}, multi {len(qa2)}, tier2 {len(qa3)}")

# --- Corpus + dense embeddings (large, cached) ---
idx = pickle.load(open(BATCH / "tier2_index.pkl", "rb"))
chunks = idx["chunks"]
embs = pickle.load(open(BATCH / "tier2_index_large.pkl", "rb"))["embeddings"]
embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
print(f"Chunks: {len(chunks)}, dense dims: {embs.shape[1]}")

# --- BM25 (compact implementation) ---
def tokenize(t):
    return re.findall(r"[a-z0-9]+", t.lower())

tokenized = [tokenize(c) for c in chunks]
doc_len = np.array([len(t) for t in tokenized], dtype=float)
avgdl = doc_len.mean()
N = len(chunks)
df = defaultdict(int)
for toks in tokenized:
    for w in set(toks):
        df[w] += 1
idf = {w: np.log(1 + (N - c + 0.5) / (c + 0.5)) for w, c in df.items()}
tf_maps = []
for toks in tokenized:
    m = defaultdict(int)
    for w in toks:
        m[w] += 1
    tf_maps.append(m)

def bm25_scores(query, k1=1.5, b=0.75):
    qw = tokenize(query)
    scores = np.zeros(N)
    for w in qw:
        if w not in idf:
            continue
        for i, m in enumerate(tf_maps):
            f = m.get(w, 0)
            if f:
                scores[i] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * doc_len[i] / avgdl))
    return scores

# --- Question embeddings (large) ---
import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.embedder import OpenRouterEmbedder
embedder = OpenRouterEmbedder()

def evaluate(qa, label):
    questions = [q for q, _ in qa]
    q_embs = embedder.embed(questions)
    out = {"dense": {1: 0, 5: 0, 10: 0}, "bm25": {1: 0, 5: 0, 10: 0},
           "hybrid": {1: 0, 5: 0, 10: 0}}
    for qi, (q, gold) in enumerate(qa):
        dense_sims = (q_embs[qi] @ embs_n.T)
        dense_rank = np.argsort(-dense_sims)
        bm = bm25_scores(q)
        bm_rank = np.argsort(-bm)
        # RRF fusion (k=60)
        rrf = np.zeros(N)
        for rank, doc in enumerate(dense_rank):
            rrf[doc] += 1.0 / (60 + rank + 1)
        for rank, doc in enumerate(bm_rank):
            rrf[doc] += 1.0 / (60 + rank + 1)
        hyb_rank = np.argsort(-rrf)
        for name, ranking in (("dense", dense_rank), ("bm25", bm_rank), ("hybrid", hyb_rank)):
            for k in (1, 5, 10):
                top = [chunks[i] for i in ranking[:k]]
                if all(any(g.lower() in c.lower() for c in top) for g in gold):
                    out[name][k] += 1
    n = len(qa)
    print(f"\n[{label}] n={n}")
    for name in ("dense", "bm25", "hybrid"):
        print(f"  {name}: " + "  ".join(f"@{k} {out[name][k]/n:.3f}" for k in (1, 5, 10)))
    return out

r1 = evaluate(qa1, "Single-hop (Tier 1)")
r2 = evaluate(qa2, "Multi-hop")
r3 = evaluate(qa3, "Tier 2 facts")

n_tot = len(qa1) + len(qa2) + len(qa3)
print(f"\n[ALL n={n_tot}] graph alone = {(len(qa1)+len(qa2))/n_tot:.3f}")
for name in ("dense", "bm25", "hybrid"):
    for k in (5, 10):
        tot = r1[name][k] + r2[name][k] + r3[name][k]
        hyb_graph = (len(qa1) + len(qa2) + r3[name][k]) / n_tot
        print(f"  {name}@{k}: {tot/n_tot:.3f} | graph+{name}@{k}: {hyb_graph:.3f}")
