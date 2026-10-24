"""RAG embedder A/B — same frozen QA, MiniLM baseline vs a larger embedder.

Motivation (2026-10-12): rag_eval.py showed vector recall is the weak arm
vs graph traversal, and its index (tier2_index.pkl) is all-MiniLM-L6-v2.
Question: how much of the gap is the embedder, not the approach?

Design rules (do not weaken):
- QA sets are built ONCE (seed 42, same construction as rag_eval.py) and
  frozen to results/rag_qa_seed42.json; both arms answer the SAME sets.
- MiniLM arm reuses the pipeline artifact embeddings (no re-embedding).
- New arm re-embeds all 25,889 Tier 2 chunks with the chosen model.
- bge models get the standard query instruction prefix on queries only.

Backends:
  --backend fastembed --model BAAI/bge-base-en-v1.5   (local, ~136 min
      measured 2026-10-12 on this 2-core VM for the full corpus)
  --backend fastembed --model BAAI/bge-large-en-v1.5  (local, slower)
  --backend openrouter --model openai/text-embedding-3-large
      (~7.4M tokens for the corpus, ~$1; needs a ROTATED OPENROUTER_API_KEY)

Usage:
  python rag_eval_embedder.py --backend fastembed --model BAAI/bge-base-en-v1.5
  python rag_eval_embedder.py --smoke   # 300-chunk plumbing check only
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BATCH = Path("batches/jfk-hybrid-v1")
QA_PATH = Path("results/rag_qa_seed42.json")
OUT_PATH = Path("results/rag_embedder_ab.json")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_qa() -> dict:
    """Identical construction/sampling order to rag_eval.py (seed 42)."""
    if QA_PATH.exists():
        return json.loads(QA_PATH.read_text())
    random.seed(42)
    tier1, tier2 = [], []
    for line in open(BATCH / "routed.jsonl"):
        r = json.loads(line)
        t = (r["subject"], r["relation"], r["object"])
        (tier1 if r.get("tier") == 1 else tier2 if r.get("tier") == 2
         else []).append(t)
    tier1 = list(set(tier1))
    tier2 = list(set(tier2))
    t1_verbs = [t for t in tier1 if t[1] != "associated_with"]
    t2_verbs = [t for t in tier2 if t[1] != "associated_with"]

    qa1 = [(f"What did {s} {rel}?", [o])
           for s, rel, o in random.sample(t1_verbs, 60)]

    by_subj = defaultdict(list)
    for s, rel, o in tier1:
        by_subj[s.lower()].append((s, rel, o))
    paths = []
    for s, rel, o in tier1:
        for s2, rel2, o2 in by_subj.get(o.lower(), []):
            if o2.lower() != s.lower():
                paths.append((s, rel, o, rel2, o2))
    paths = random.sample(paths, min(40, len(paths)))
    qa2 = [(f"What is connected to {s} through {mid}?", [mid, o2])
           for s, _r1, mid, _r2, o2 in paths]

    qa3 = [(f"What did {s} {rel}?", [o])
           for s, rel, o in random.sample(t2_verbs, min(60, len(t2_verbs)))]

    qa = {"single_hop_tier1": qa1, "multi_hop": qa2, "tier2_facts": qa3}
    QA_PATH.write_text(json.dumps(qa, ensure_ascii=False, indent=1))
    return qa


def evaluate(encode_q, embs_n, chunks, qa, query_prefix=""):
    def run(qa_list):
        out = {}
        for k in (1, 5, 10):
            out[k] = 0
        for q, gold in qa_list:
            qe = encode_q([query_prefix + q])
            qe = qe / (np.linalg.norm(qe, axis=1, keepdims=True) + 1e-9)
            top_idx = np.argsort(-(qe @ embs_n.T)[0])
            for k in (1, 5, 10):
                top = [chunks[i] for i in top_idx[:k]]
                if all(any(g.lower() in c.lower() for c in top) for g in gold):
                    out[k] += 1
        n = len(qa_list)
        return {f"@{k}": round(v / n, 3) for k, v in out.items()} | {"n": n}

    return {name: run(qa_list) for name, qa_list in qa.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fastembed", "openrouter"],
                    default="fastembed")
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--smoke", action="store_true",
                    help="embed only 300 chunks; plumbing check, recall invalid")
    args = ap.parse_args()

    qa = build_qa()
    idx = pickle.load(open(BATCH / "tier2_index.pkl", "rb"))
    chunks = idx["chunks"]
    if args.smoke:
        chunks = chunks[:300]

    results = {"qa_file": str(QA_PATH), "smoke": args.smoke, "arms": {}}

    # --- Arm A: MiniLM from the pipeline artifact (queries via MiniLM)
    from sentence_transformers import SentenceTransformer
    mini = SentenceTransformer(idx["model"])
    embs = idx["embeddings"]
    if args.smoke:
        embs = embs[:300]
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    results["arms"]["minilm_l6_v2"] = evaluate(
        lambda qs: mini.encode(qs, convert_to_numpy=True),
        embs_n, chunks, qa)
    print("MiniLM arm:", json.dumps(results["arms"]["minilm_l6_v2"]))

    # --- Arm B: candidate embedder over the same chunks
    t0 = time.time()
    if args.backend == "fastembed":
        from fastembed import TextEmbedding
        emb_model = TextEmbedding(args.model)
        new_embs = np.array(list(emb_model.embed(chunks)), dtype=np.float32)
        def encode_q(qs):
            return np.array(list(emb_model.embed(qs)),
                            dtype=np.float32)
    else:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from pipeline.embedder import OpenRouterEmbedder
        emb_model = OpenRouterEmbedder(args.model)
        new_embs = emb_model.embed(chunks)
        encode_q = emb_model.embed
    embed_secs = round(time.time() - t0, 1)
    new_n = new_embs / (np.linalg.norm(new_embs, axis=1, keepdims=True) + 1e-9)
    prefix = BGE_QUERY_PREFIX if "bge" in args.model.lower() else ""
    arm_name = f"{args.backend}:{args.model}"
    results["arms"][arm_name] = evaluate(encode_q, new_n, chunks, qa,
                                         query_prefix=prefix)
    results["arms"][arm_name]["embed_secs"] = embed_secs
    results["arms"][arm_name]["dims"] = int(new_embs.shape[1])
    print(f"{arm_name}:", json.dumps(results["arms"][arm_name]))

    if not args.smoke:
        OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print("saved:", OUT_PATH)
    else:
        print("smoke run — results NOT saved, recall numbers are meaningless")


if __name__ == "__main__":
    main()
