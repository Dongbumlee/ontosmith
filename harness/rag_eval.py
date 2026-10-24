"""RAG vs Graph RAG evaluation v2 — jfk-hybrid-v1 only, honest design.

Three question types (2026-10-12):
1. Single-hop: from this batch's Tier 1 verb triples.
   Graph hit is 100% by construction; vector recall@k is the signal.
2. Multi-hop: composed A->B->C paths in the Tier 1 graph.
   Question asks A..C; graph traverses, vector must surface BOTH facts.
3. Tier 2 facts: triples routed to Tier 2 (not in graph).
   Graph hit is 0% by construction; vector recall is the signal.

Vector index: pipeline artifact tier2_index.pkl (MiniLM, Tier 2 docs).
"""
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

random.seed(42)
BATCH = Path("batches/jfk-hybrid-v1")

# --- Load this batch's routed triples
tier1, tier2 = [], []
for l in open(BATCH / "routed.jsonl"):
    r = json.loads(l)
    t = (r["subject"], r["relation"], r["object"])
    (tier1 if r.get("tier") == 1 else tier2 if r.get("tier") == 2 else []).append(t)
tier1 = list(set(tier1)); tier2 = list(set(tier2))
t1_verbs = [t for t in tier1 if t[1] != "associated_with"]
t2_verbs = [t for t in tier2 if t[1] != "associated_with"]
print(f"Tier 1: {len(tier1)} (verbs {len(t1_verbs)}), Tier 2: {len(tier2)} (verbs {len(t2_verbs)})")

graph = defaultdict(set)
for s, rel, o in tier1:
    graph[s.lower()].add((rel.lower(), o.lower()))
graph_full = defaultdict(set)
for s, rel, o in tier1:
    graph_full[(s.lower(), rel.lower())].add(o.lower())

# --- Vector index (pipeline artifact) + embedder choice
import os

idx = pickle.load(open(BATCH / "tier2_index.pkl", "rb"))
chunks = idx["chunks"]
USE_LARGE = os.environ.get("RAG_EMBEDDER") == "large"
if USE_LARGE:
    from pipeline.embedder import OpenRouterEmbedder
    large = OpenRouterEmbedder()
    cache_f = BATCH / "tier2_index_large.pkl"
    if cache_f.exists():
        embs = pickle.load(open(cache_f, "rb"))["embeddings"]
        print("Loaded cached large embeddings")
    else:
        print(f"Embedding {len(chunks)} chunks with text-embedding-3-large...")
        embs = large.embed(chunks)
        pickle.dump({"embeddings": embs, "model": "openai/text-embedding-3-large"},
                    open(cache_f, "wb"))
    model = None
else:
    embs = idx["embeddings"]
    model = SentenceTransformer(idx["model"])
embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
print(f"Vector index: {len(chunks)} chunks, embedder: {'text-embedding-3-large' if USE_LARGE else idx['model']}")

def embed_q(text):
    if USE_LARGE:
        return large.embed([text])
    q = model.encode([text], convert_to_numpy=True)
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)

def vec_eval(qa, ks=(1, 5, 10)):
    out = {k: 0 for k in ks}
    for q, gold in qa:
        qe = embed_q(q)
        sims = (qe @ embs_n.T)[0]
        top_idx = np.argsort(-sims)
        for k in ks:
            top = [chunks[i] for i in top_idx[:k]]
            if all(any(g.lower() in c.lower() for c in top) for g in gold):
                out[k] += 1
    return out

# --- Type 1: single-hop from Tier 1 verbs
qa1 = [(f"What did {s} {rel}?", [o]) for s, rel, o in random.sample(t1_verbs, 60)]
r1 = vec_eval(qa1)
print("\n[Single-hop, Tier 1 facts] n=60 (graph = 1.000 by construction)")
for k, h in r1.items():
    print(f"  vector@{k}: {h}/60 = {h/60:.3f}")

# --- Type 2: multi-hop A->B->C
by_subj = defaultdict(list)
for s, rel, o in tier1:
    by_subj[s.lower()].append((s, rel, o))
paths = []
for s, rel, o in tier1:
    for s2, rel2, o2 in by_subj.get(o.lower(), []):
        if o2.lower() != s.lower():
            paths.append((s, rel, o, rel2, o2))
print(f"\nMulti-hop paths found: {len(paths)}")
paths = random.sample(paths, min(40, len(paths)))
qa2 = [(f"What is connected to {s} through {mid}?", [mid, o2]) for s, r1_, mid, r2_, o2 in paths]
r2 = vec_eval(qa2)
# Graph traversal: A -[*2]-> C exists by construction for sampled paths
print(f"[Multi-hop A->B->C] n={len(paths)} (graph traversal = 1.000 by construction)")
for k, h in r2.items():
    print(f"  vector@{k} (both facts): {h}/{len(paths)} = {h/len(paths):.3f}")

# --- Type 3: Tier 2 facts (not in graph)
qa3 = [(f"What did {s} {rel}?", [o]) for s, rel, o in random.sample(t2_verbs, min(60, len(t2_verbs)))]
r3 = vec_eval(qa3)
print(f"\n[Tier 2 facts, graph = 0 by construction] n={len(qa3)}")
for k, h in r3.items():
    print(f"  vector@{k}: {h}/{len(qa3)} = {h/len(qa3):.3f}")
