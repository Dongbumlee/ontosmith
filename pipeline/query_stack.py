"""Query stack — the production
retrieval path over the GraphStore serving index.

One code path serves production queries AND the pinned-QA eval
arms (§7 contract, gap G-20): the arms below are flags over the
same retrievers, so measurement and production cannot drift apart.

Components:
  Chunker            the §6 chunking interface; the measured
                     baseline is FixedWindowChunker (1000-char
                     windows, identical to cuad_eval.py / stage_index)
  project_serving()  E5 serving projection for the judged CUAD full
                     corpus: routed_v6 triples -> registry-resolved
                     Entity/REL (minted IDs, corpus 'cuad'), and
                     ALL 509 contracts' chunks (cached embeddings
                     reused, zero re-embedding) -> :Chunk nodes with
                     HAS_CHUNK + MENTIONS (G-15 second half: Tier 1
                     documents' chunks are indexed too, tagged by
                     their best fact tier)
  Linker             deterministic question -> registry entities
                     (name index spans + find_candidates + Lucene
                     full-text over Entity.name), scored, top-k
  retrievers         dense / hybrid (vector + Lucene, RRF) over
                     GraphStore.search_chunks; graph-current (the
                     cuad_eval token-overlap arm ported onto the
                     store snapshot); graph-PPR (OSS borrow #2,
                     HippoRAG 2: Personalized PageRank over the
                     entity graph, seeded by linker hits and
                     query-to-triple embedding matches, Tier 1
                     edges at full weight, Tier 2 at 0.5 — fixed
                     ex ante; chunks participate via MENTIONS)
  Router             OSS borrow #4 (LightRAG dual-level): one cheap
                     Haiku call splits a question into entity
                     keywords + theme keywords + a route class;
                     every split is logged (auditable)
  Answerer           OSS borrow #7 (named evidence-packing recipes
                     per route class under a fixed token budget)
                     + #13 (nano-graphrag fail_response discipline):
                     answers carry (doc_id, span) citations that
                     resolve to source text; abstention is a
                     first-class output under an ex-ante evidence
                     floor — never a fabricated answer.

CLI:
  python -m pipeline.query_stack project   build the serving data
  python -m pipeline.query_stack measure   latency + A/B over QA 116
  python -m pipeline.query_stack smoke     answerer smoke, 12 questions
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np

from . import connections as _connections
from .reference_profile import CHAT_URL as _REF_CHAT_URL
from .reference_profile import EMBED_MODEL as _REF_EMBED_MODEL
from .reference_profile import PROPOSER_MODEL as _REF_PROPOSER_MODEL
from .reference_profile import PROVIDER_SORT as _REF_PROVIDER_SORT

ROOT = Path(__file__).resolve().parent.parent
BATCH_ID = "cuad-legal-full"
RUN_ID = "p3b-v6"
CORPUS = "cuad"
BATCH_DIR = ROOT / "batches" / BATCH_ID
EVAL_DIR = BATCH_DIR / "eval"

EMBED_MODEL = _REF_EMBED_MODEL
CHAT_MODEL = _REF_PROPOSER_MODEL
CHAT_URL = _REF_CHAT_URL

# --- ex-ante constants (fixed before measurement; reported as-is) --
TIER2_EDGE_DISCOUNT = 0.5     # PPR + graph arms (mirrors cuad_eval)
PPR_ALPHA = 0.85              # PageRank damping
PPR_TOP_TRIPLES = 10          # query-to-triple seeds (HippoRAG 2)
ABSTAIN_FLOOR = 0.35          # top-1 dense chunk cosine floor
EVIDENCE_TOKEN_BUDGET = 6000  # answerer evidence budget (chars/4)
CHUNK_FETCH_K = 400           # chunks fetched per index arm / question

# ----------------------------------------------------------------------
# tokenization / fusion — mirrors cuad_eval.py exactly (the A/B in
# scripts + P3B_REPORT validates the mirror against the v6 numbers)
# ----------------------------------------------------------------------
STOP = set("what which contracts contract corpus agreement agreements is are does do "
           "in of the a an and or to for with without by are there it its each every "
           "all list give state how many among across where when into from as at on "
           "used throughout refer party".split())


def tokenize(t: str) -> list[str]:
    """Word tokens for the retrieval arms. Script-aware
    (issue #53): the pre-fix pattern ``[a-z0-9]+`` yielded
    ZERO tokens for a Korean-only query, so the lexical arm
    never fired and the graph arm's overlap sets were empty
    for non-Latin content. ``[^\\W_]`` is the Unicode word
    class minus underscore: on lowercased ASCII input it
    produces exactly the old ``[a-z0-9]+`` stream (the
    cuad_eval mirror the flagship numbers rest on is
    unchanged — pinned by test), while Hangul / CJK / kana /
    Cyrillic / accented-Latin runs now tokenize on
    whitespace-and-punctuation boundaries like Latin text.

    Known limit, stated plainly: this is whitespace
    tokenization, not morphological analysis — a Korean
    particle attached to a noun (삼성전자는) does not match
    the bare noun (삼성전자). Queries phrased with bare
    entity surfaces match; full morphology needs an
    analyzer the stock stack does not bundle (see the
    fulltext analyzer knob in graph_store)."""
    return re.findall(r"[^\W_]+", (t or "").lower())


def is_content_token(tok: str) -> bool:
    """Whether a token can carry retrieval signal. The
    historical rule was ``len(tok) > 2`` — an English-shaped
    filter that drops function words (a / of / to). Applied
    to Hangul it also drops dense two-character content
    words (본사, 위치, 서울), which gutted the arms even
    after tokenization was fixed; so a two-character token
    counts when it is non-ASCII. ASCII behavior is exactly
    the old rule."""
    return len(tok) > 2 or (len(tok) == 2 and not tok.isascii())


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    score: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, d in enumerate(r):
            score[d] += 1.0 / (k + i + 1)
    return [d for d, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def docs_from_chunk_rows(rows: list[dict]) -> list[str]:
    """Doc ranking by max chunk score (cuad_eval aggregation)."""
    best: dict[str, float] = {}
    for r in rows:
        d = r["doc_id"]
        if r["score"] > best.get(d, -1e18):
            best[d] = r["score"]
    return [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]


# ----------------------------------------------------------------------
# G-16: Chunker interface (§6). Chunking policy lives behind this
# protocol so a semantic chunker can A/B against the pinned QA
# without editing any stage.
# ----------------------------------------------------------------------
class Chunker(Protocol):
    def chunk(self, text: str) -> list[tuple[int, str]]:
        """Return [(chunk_idx, chunk_text)] for one document."""
        ...


class FixedWindowChunker:
    """The measured baseline: 1000-char windows, chunks whose
    stripped length is <= 100 chars are dropped. Byte-identical to
    cuad_eval.py's chunk universe and stage_index's windows."""

    def __init__(self, size: int = 1000, min_chars: int = 100):
        self.size = size
        self.min_chars = min_chars

    def chunk(self, text: str) -> list[tuple[int, str]]:
        out = []
        for i in range(0, len(text), self.size):
            ch = text[i:i + self.size]
            if len(ch.strip()) > self.min_chars:
                out.append((i // self.size, ch))
        return out


# ----------------------------------------------------------------------
# chat helper (mirrors pipeline/proposer.py's pattern: Haiku takes no
# temperature; usage incl. cost is returned for accounting)
# ----------------------------------------------------------------------
def _chat(content: str, max_tokens: int, key: str,
          model: str = CHAT_MODEL) -> tuple[str, dict]:
    _binding = _connections.seat("proposer")
    if not key:
        raise _connections.BindingKeyError(
            f"seat 'proposer' cannot run: its key variable "
            f"{_binding.key_env} is not set")
    import requests
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    # provider.sort is an OpenRouter-dialect field (issue
    # #41): emitted only when the seat's dialect declares it.
    if _binding.is_openrouter_dialect:
        payload["provider"] = {"sort": _REF_PROVIDER_SORT}
    if not _binding.has_capability(
            _connections.CAP_NO_TEMPERATURE, model):
        payload["temperature"] = 0.0
    text = ""
    data = {}
    for _attempt in range(2):  # Haiku occasionally returns null content
        r = requests.post(_binding.chat_url,
                          headers={"Authorization": f"Bearer {key}"},
                          json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"].get("content")
        if content:
            text = content.strip()
            break
    norm = _connections.normalize_usage(data.get("usage"), _binding)
    usage_out = {
        "prompt_tokens": norm["input_tokens"],
        "completion_tokens": norm["output_tokens"],
        "cost": norm["cost"],
        "cost_basis": norm["cost_basis"],
    }
    if not text:
        return "", usage_out
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```"
                         else lines[1:])
    return text, usage_out


def _chat_cost(usage: dict) -> float | None:
    """The call's metered cost, or None when the provider
    reported none and no user rates are declared (issue
    #41). The pre-generalization fallback priced a missing
    cost from hardcoded list rates for one model on one
    provider — an invented figure, now deleted: unknown
    stays unknown."""
    cost = usage.get("cost")
    return None if cost is None else float(cost)


# ----------------------------------------------------------------------
# E5 serving projection for the judged full corpus
# ----------------------------------------------------------------------
def project_serving(gs=None, registry=None) -> dict:
    """Project batches/cuad-legal-full/routed_v6_judged.jsonl +
    the cached all-chunk embeddings into the local store.

    Entities resolve through the registry name index (corpus
    'cuad'); unresolved surfaces are registered (minted IDs) —
    registry-first (§4), the graph is its projection. Chunks come
    from eval/chunk_index_all.pkl (the eval's own cache — the same
    embeddings, zero re-embedding spend) and are verified against
    FixedWindowChunker over texts.jsonl before being written.
    """
    import pickle

    from .file_registry import SqliteEntityRegistry
    from .graph_store import GraphStore, chunk_id_for
    from .registry import Entity

    gs = gs or GraphStore()
    registry = registry or SqliteEntityRegistry(
        ROOT / "data" / "registry.db")
    gs.ensure_schema()

    texts: dict[str, str] = {}
    source_path: dict[str, str] = {}
    for line in open(BATCH_DIR / "texts.jsonl"):
        r = json.loads(line)
        texts[r["doc_id"]] = r["text"]
        source_path[r["doc_id"]] = r.get("source_path", "")

    routed = [json.loads(l) for l in
              open(BATCH_DIR / "routed_v6_judged.jsonl")]

    # --- entity resolution through the registry name index --------
    resolved: dict[str, Entity] = {}
    n_existing = n_minted = n_surface_diff = 0
    surfaces = sorted({r["subject"] for r in routed}
                      | {r["object"] for r in routed})
    for name in surfaces:
        ent = registry.resolve(name, corpus=CORPUS)
        if ent is None:
            eid = registry.create(Entity(
                entity_id="", name=name, label="", aliases=[],
                batch_id=BATCH_ID, run_id=RUN_ID, doc_id="",
                corpus=CORPUS))
            ent = registry.get(eid)
            n_minted += 1
        else:
            n_existing += 1
        if ent.name != name:
            n_surface_diff += 1
        resolved[name] = ent
    n_ent = gs.upsert_entities(
        [{"entity_id": e.entity_id, "name": e.name,
          "aliases": e.aliases, "corpus": CORPUS}
         for e in resolved.values()], corpus=CORPUS)

    # --- REL edges (GraphStore write path; valid_at/invalid_at
    #     written as null — schema support only, borrow #6) --------
    edge_rows = []
    sidecar = []
    for r in routed:
        s, o = resolved[r["subject"]], resolved[r["object"]]
        edge_rows.append({
            "subject_id": s.entity_id, "object_id": o.entity_id,
            "relation": r["relation"], "batch_id": BATCH_ID,
            "run_id": RUN_ID, "doc_id": r["doc_id"],
            "p_true": r["p_true"], "tier": r["tier"],
            "tag": "VERIFIED" if r["tier"] == 1 else "UNVERIFIED",
            "model": r.get("method", ""), "gate": "jev-1.13",
            "surface_s": r["subject"], "surface_o": r["object"],
            "quote": r.get("quote", ""),
            "valid_at": None, "invalid_at": None})
        sidecar.append({
            "s_id": s.entity_id, "o_id": o.entity_id,
            "type": r["relation"], "doc_id": r["doc_id"],
            "surface_s": r["subject"], "surface_o": r["object"],
            "canonical_s": s.name, "canonical_o": o.name,
            "quote": r.get("quote", ""),
            "p_true": r["p_true"], "tier": r["tier"]})
    n_edges = gs.upsert_edges(edge_rows)
    with open(EVAL_DIR / "p3b_surface_map.jsonl", "w") as f:
        for row in sidecar:
            f.write(json.dumps(row) + "\n")

    # --- chunks: verify the cache against the Chunker, then write
    cache = pickle.load(open(EVAL_DIR / "chunk_index_all.pkl", "rb"))
    meta, chunks, embs = cache["meta"], cache["chunks"], cache["embeddings"]
    chunker = FixedWindowChunker()
    by_doc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for m, ch in zip(meta, chunks):
        by_doc[m["doc_id"]].append((int(m["chunk_idx"]), ch))
    n_mismatch = 0
    for doc_id, text in texts.items():
        want = chunker.chunk(text)
        got = sorted(by_doc.get(doc_id, []))
        if [i for i, _ in want] != [i for i, _ in got] or \
                any(a != b for (_, a), (_, b) in zip(want, got)):
            n_mismatch += 1
    if n_mismatch:
        raise RuntimeError(
            f"chunk cache diverges from FixedWindowChunker on "
            f"{n_mismatch} docs — refusing to project a stale cache")

    doc_best_tier: dict[str, int] = {}
    for r in routed:
        cur = doc_best_tier.get(r["doc_id"], 2)
        doc_best_tier[r["doc_id"]] = min(cur, int(r["tier"]))

    doc_rows = [{"doc_id": d, "source_path": source_path.get(d, ""),
                 "corpus": CORPUS} for d in sorted(texts)]
    for i in range(0, len(doc_rows), 500):
        gs.c.run([{
            "statement":
                "UNWIND $rows AS row "
                "MERGE (d:Document {doc_id: row.doc_id}) "
                "SET d.source_path = row.source_path, "
                "d.corpus = row.corpus",
            "parameters": {"rows": doc_rows[i:i + 500]}}])

    emb_by_key = {(m["doc_id"], int(m["chunk_idx"])): embs[i]
                  for i, m in enumerate(meta)}
    chunk_rows = []
    for m, ch in zip(meta, chunks):
        doc_id, idx = m["doc_id"], int(m["chunk_idx"])
        tier = doc_best_tier.get(doc_id, 2)
        chunk_rows.append({
            "chunk_id": chunk_id_for(BATCH_ID, doc_id, idx),
            "doc_id": doc_id, "idx": idx, "text": ch, "tier": tier,
            "tag": "VERIFIED" if tier == 1 else "UNVERIFIED",
            "batch_id": BATCH_ID, "corpus": CORPUS,
            "embedding": [float(x) for x in
                          emb_by_key[(doc_id, idx)]]})
    n_chunks = 0
    for i in range(0, len(chunk_rows), 50):
        n_chunks += gs.c.one(
            "UNWIND $rows AS row "
            "MERGE (c:Chunk {chunk_id: row.chunk_id}) "
            "SET c.doc_id = row.doc_id, c.idx = row.idx, "
            "c.text = row.text, c.tier = row.tier, "
            "c.tag = row.tag, c.batch_id = row.batch_id, "
            "c.corpus = row.corpus, c.embedding = row.embedding "
            "WITH row, c "
            "MATCH (d:Document {doc_id: row.doc_id}) "
            "MERGE (d)-[:HAS_CHUNK]->(c) RETURN count(c)",
            {"rows": chunk_rows[i:i + 50]}) or 0

    # --- MENTIONS: a chunk mentions an entity when a surface form
    #     of one of the doc's own triple endpoints is attested in
    #     the chunk text (verbatim, then case-insensitive)
    ent_surfaces: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))  # doc_id -> entity_id -> surfaces
    for r in routed:
        for surf_key, name in (("subject", r["subject"]),
                               ("object", r["object"])):
            ent = resolved[name]
            ent_surfaces[r["doc_id"]][ent.entity_id].add(name)
            ent_surfaces[r["doc_id"]][ent.entity_id].add(ent.name)
            for a in ent.aliases or []:
                ent_surfaces[r["doc_id"]][ent.entity_id].add(a)
    pairs: set[tuple[str, str]] = set()
    for m, ch in zip(meta, chunks):
        doc_id = m["doc_id"]
        cid = chunk_id_for(BATCH_ID, doc_id, int(m["chunk_idx"]))
        low = ch.lower()
        for eid, surfs in ent_surfaces.get(doc_id, {}).items():
            if any(s in ch or s.lower() in low for s in surfs if s):
                pairs.add((cid, eid))
    pair_rows = [{"chunk_id": c, "eid": e} for c, e in sorted(pairs)]
    n_mentions = 0
    for i in range(0, len(pair_rows), 1000):
        n_mentions += gs.c.one(
            "UNWIND $rows AS row "
            "MATCH (c:Chunk {chunk_id: row.chunk_id}), "
            "(e:Entity {entity_id: row.eid}) "
            "MERGE (c)-[:MENTIONS]->(e) RETURN count(*)",
            {"rows": pair_rows[i:i + 1000]}) or 0

    gs.c.run([{"statement": "CALL db.awaitIndexes(300)",
               "parameters": {}}])
    result = {
        "batch_id": BATCH_ID, "run_id": RUN_ID, "corpus": CORPUS,
        "surfaces": len(surfaces), "entities_written": int(n_ent),
        "resolved_existing": n_existing, "minted": n_minted,
        "surface_differs_from_canonical": n_surface_diff,
        "edges_written": int(n_edges), "documents": len(doc_rows),
        "chunks": int(n_chunks), "mentions": int(n_mentions),
        "chunk_cache_verified_against_chunker": True,
    }
    (EVAL_DIR / "p3b_projection.json").write_text(
        json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------
# Entity linker — deterministic, scored (§7 steps 1/2/4: normalized
# name + alias spans over the registry name index, registry
# find_candidates, Lucene full-text over Entity.name)
# ----------------------------------------------------------------------
class Linker:
    def __init__(self, registry, gs):
        from .registry import index_normalize
        self.registry = registry
        self.gs = gs
        self._norm = index_normalize
        ents = [e for e in registry.all_entities()
                if e.corpus == CORPUS]
        self.entities = {e.entity_id: e for e in ents}
        # (normalized key, key tokens, entity_id, kind); tokens are
        # alphanumeric runs so punctuation/quotes/hyphens in either
        # the name or the question cannot break a span match
        self._name_keys = []
        for e in ents:
            k = index_normalize(e.name)
            if k:
                self._name_keys.append(
                    (k, re.findall(r"[a-z0-9]+", k), e.entity_id,
                     "name"))
            for a in e.aliases or []:
                ka = index_normalize(a)
                if ka:
                    self._name_keys.append(
                        (ka, re.findall(r"[a-z0-9]+", ka), e.entity_id,
                         "alias"))

    def link(self, question: str, k: int = 10) -> list[dict]:
        """Top-k linked entities: [{entity_id, name, score, via}].
        Scores: exact normalized-name span 100, alias span 90,
        registry find_candidates score (its own 0..100 scale),
        full-text hits scaled to <=50. Deterministic ordering."""
        qkey = self._norm(question)
        qtoks = re.findall(r"[a-z0-9]+", qkey)
        best: dict[str, tuple[float, str]] = {}

        def offer(eid: str, score: float, via: str) -> None:
            if eid in self.entities and \
                    score > best.get(eid, (-1.0, ""))[0]:
                best[eid] = (score, via)

        for key, ktoks, eid, kind in self._name_keys:
            # token-span containment (word boundaries): a raw
            # substring check false-positives badly ('ovid' inside
            # 'provide', 'vert' inside 'advertising')
            if len(key) < 4 or not ktoks or len(ktoks) > len(qtoks):
                continue
            n = len(ktoks)
            if any(qtoks[i:i + n] == ktoks
                   for i in range(len(qtoks) - n + 1)):
                offer(eid, 100.0 if kind == "name" else 90.0,
                      f"{kind}-span")
        for cand in self.registry.find_candidates(
                question, limit=5, corpus=CORPUS):
            from .file_registry import SqliteEntityRegistry
            offer(cand.entity_id,
                  SqliteEntityRegistry._score(
                      cand, self._norm(question),
                      set(self._norm(question).split()),
                      question.lower()),
                  "find_candidates")
        rows = self.gs.c.run([{
            "statement":
                "CALL db.index.fulltext.queryNodes($index, $q) "
                "YIELD node, score RETURN node.entity_id, score "
                "ORDER BY score DESC LIMIT 20",
            "parameters": {"index": "entity_name_ft",
                           "q": " OR ".join(
                               t for t in tokenize(question)
                               if len(t) > 2) or question}}])[0]
        for r in rows:
            eid, sc = r["row"]
            offer(eid, min(50.0, float(sc) * 10.0), "fulltext")
        key_len: dict[str, int] = {}
        for k2, _kt, eid, _kind in self._name_keys:
            key_len[eid] = max(key_len.get(eid, 0), len(k2))
        out = [{"entity_id": eid, "name": self.entities[eid].name,
                "score": sc, "via": via}
               for eid, (sc, via) in best.items() if sc > 0]
        # ties break toward the LONGER (more specific) name key,
        # then name/id for determinism
        out.sort(key=lambda d: (-d["score"], -key_len.get(
            d["entity_id"], 0), d["name"], d["entity_id"]))
        return out[:k]


# ----------------------------------------------------------------------
# Graph snapshot — the graph arms read the store once through
# GraphStore's client and score in memory (the snapshot IS the
# production retriever's working set; its load time is reported
# separately in measurement, never hidden inside per-query latency)
# ----------------------------------------------------------------------
def load_attestations(batch_dir: Path, eval_dir: Path):
    """Per-attestation rows for the graph arms, from the projection
    record (the surface-map rows).

    Each surface-map row carries the (p_true, tier) the projection
    recorded for THAT attestation, and those values are the ones
    used. The previous implementation joined
    the surface rows to the judged routed rows through a dict keyed
    (subject, object, relation, doc_id) — duplicate keys silently
    overwrote each other (the last row won), and any surface row
    whose key missed the dict silently scored the default
    (0.3, 2). The per-attestation join here was proven equivalent
    on the frozen artifacts before it landed: per-question
    recall@20/@5 for both graph arms is unchanged
    (0.6516 / 0.7115 overall).

    Returns (attestations, att_tokens, stats). stats counts the
    surface rows whose key has no judged counterpart
    ("unmatched": their own projected values are used — the count
    is reported in the eval output, never silently defaulted) and
    rows carrying no score of their own and no judged counterpart
    ("unscored": excluded from the arms, and counted).
    """
    judged: dict[tuple, list] = {}
    for line in open(batch_dir / "routed_v6_judged.jsonl"):
        r = json.loads(line)
        judged.setdefault(
            (r["subject"], r["object"], r["relation"], r["doc_id"]),
            []).append((float(r["p_true"]), int(r["tier"])))
    attestations: list[dict] = []
    att_tokens: list[set] = []
    unmatched = unscored = 0
    for line in open(eval_dir / "p3b_surface_map.jsonl"):
        s = json.loads(line)
        key = (s["surface_s"], s["surface_o"], s["type"],
               s["doc_id"])
        if s.get("p_true") is not None and s.get("tier") is not None:
            p_true, tier = float(s["p_true"]), int(s["tier"])
            if key not in judged:
                unmatched += 1
        elif key in judged:
            p_true, tier = judged[key][0]
        else:
            unscored += 1
            continue
        att = {"s_id": s["s_id"], "o_id": s["o_id"],
               "type": s["type"], "doc_id": s["doc_id"],
               "surface_s": s["surface_s"],
               "surface_o": s["surface_o"],
               "quote": s.get("quote", ""),
               "p_true": p_true, "tier": tier}
        attestations.append(att)
        att_tokens.append(set(tokenize(
            f"{att['surface_s']} {att['type']} "
            f"{att['surface_o']}")))
    return attestations, att_tokens, {
        "unmatched": unmatched, "unscored": unscored}


class GraphData:
    """Two views of the projected graph, both loaded from the
    serving projection:

    facts        the store's REL edges. GraphStore.upsert_edges
                 MERGEs at attestation level (key includes doc_id +
                 att_id) and stores the extractor surfaces on the
                 edge, so facts ARE the per-attestation rows and
                 the store alone answers "which documents attest
                 this fact". On a pre-fix projection the same load
                 yields the collapsed view (one edge per fact, a
                 single doc_id) — graph_collapsed_ranking measures
                 exactly that difference.
    attestations the per-(fact, doc) rows from the projection
                 record (eval/p3b_surface_map.jsonl: resolved IDs +
                 extractor surfaces + quotes), each scored with its
                 OWN recorded p_true / tier (see load_attestations;
                 the join statistics ride along as
                 ``attestation_join``). The graph arms score per
                 attestation — that is what cuad_eval's graph arm
                 iterates — while PPR structure and the linker work
                 on facts/entities.
    """

    def __init__(self, gs):
        rows = gs.c.run([{
            "statement":
                "MATCH (a:Entity)-[r:REL {batch_id: $b}]->"
                "(b:Entity) RETURN a.entity_id, a.name, "
                "b.entity_id, b.name, r.type, r.p_true, r.tier, "
                "r.doc_id, r.surface_s, r.surface_o "
                "ORDER BY r.doc_id, a.entity_id, r.type, "
                "b.entity_id",
            "parameters": {"b": BATCH_ID}}])[0]
        self.facts: list[dict] = []
        self.entity_name: dict[str, str] = {}
        for r in rows:
            v = r["row"]
            self.facts.append({
                "s_id": v[0], "o_id": v[2], "type": v[4],
                "p_true": float(v[5] or 0.3), "tier": int(v[6]),
                "doc_id": v[7],
                "surface_s": v[8], "surface_o": v[9]})
            self.entity_name[v[0]] = v[1]
            self.entity_name[v[2]] = v[3]
        (self.attestations, self.att_tokens,
         self.attestation_join) = load_attestations(BATCH_DIR,
                                                    EVAL_DIR)
        fact_surfaces: dict[tuple, tuple[str, str]] = {}
        for att in self.attestations:
            fact_surfaces.setdefault(
                (att["s_id"], att["type"], att["o_id"]),
                (att["surface_s"], att["surface_o"]))
        self.fact_tokens: list[set] = []
        for e in self.facts:
            # Post-G-28 edges carry their own attestation surfaces;
            # pre-fix edges fall back to the projection record's
            # first surface pair per fact, then to canonical names.
            if e.get("surface_s") and e.get("surface_o"):
                s_txt, o_txt = e["surface_s"], e["surface_o"]
            else:
                s_txt, o_txt = fact_surfaces.get(
                    (e["s_id"], e["type"], e["o_id"]),
                    (self.entity_name[e["s_id"]],
                     self.entity_name[e["o_id"]]))
                e["surface_s"], e["surface_o"] = s_txt, o_txt
            self.fact_tokens.append(set(tokenize(
                f"{s_txt} {e['type']} {o_txt}")))
        # chunks + mentions
        self.chunk_doc: dict[str, str] = {}
        self.doc_chunks: dict[str, list[str]] = defaultdict(list)
        rows = gs.c.run([{
            "statement":
                "MATCH (c:Chunk {batch_id: $b}) RETURN c.chunk_id, "
                "c.doc_id",
            "parameters": {"b": BATCH_ID}}])[0]
        for r in rows:
            cid, doc_id = r["row"]
            self.chunk_doc[cid] = doc_id
            self.doc_chunks[doc_id].append(cid)
        self.chunk_mentions: dict[str, list[str]] = defaultdict(list)
        rows = gs.c.run([{
            "statement":
                "MATCH (c:Chunk {batch_id: $b})-[:MENTIONS]->"
                "(e:Entity) RETURN c.chunk_id, e.entity_id",
            "parameters": {"b": BATCH_ID}}])[0]
        for r in rows:
            self.chunk_mentions[r["row"][0]].append(r["row"][1])
        # entity -> attesting docs (specificity + answerer)
        self.entity_docs: dict[str, set] = defaultdict(set)
        for e in self.attestations:
            self.entity_docs[e["s_id"]].add(e["doc_id"])
            self.entity_docs[e["o_id"]].add(e["doc_id"])
        self.n_docs = len({e["doc_id"] for e in self.attestations})


def _overlap_ranking(rows: list[dict], token_sets: list[set],
                     question: str) -> list[str]:
    qtoks = {t for t in tokenize(question)
             if t not in STOP and is_content_token(t)}
    doc_score: dict[str, float] = defaultdict(float)
    for e, ttoks in zip(rows, token_sets):
        overlap = len(qtoks & ttoks)
        if overlap >= 2:
            w = e["p_true"] * (1.0 if e["tier"] == 1
                               else TIER2_EDGE_DISCOUNT)
            doc_score[e["doc_id"]] += w * overlap
    return [d for d, _ in sorted(doc_score.items(),
                                 key=lambda kv: -kv[1])]


def graph_current_ranking(gd: GraphData, question: str) -> list[str]:
    """The cuad_eval graph arm, ported verbatim: question tokens
    (STOP removed, len > 2) vs triple tokens; overlap >= 2 scores
    p_true x tier weight x overlap into the triple's doc. Scored
    per attestation, exactly as the eval iterates routed rows."""
    return _overlap_ranking(gd.attestations, gd.att_tokens, question)


def graph_collapsed_ranking(gd: GraphData, question: str) -> list[str]:
    """The same arm scored over the store's REL edges alone (no
    projection record). On the pre-G-28 store this was the
    collapsed diagnostic (one doc per fact: 0.6316); since the
    attestation-level MERGE fix it is the store-alone serving arm
    and must reproduce the per-attestation number."""
    return _overlap_ranking(gd.facts, gd.fact_tokens, question)


# ----------------------------------------------------------------------
# graph-PPR (borrow #2, HippoRAG 2 adapted): Personalized PageRank
# over the entity graph. Edge weight = p_true x tier factor (Tier 2
# discounted 0.5, fixed ex ante). Seeds: linker hits and
# query-to-triple embedding matches (top-10 triples seed their
# endpoints), both scaled by node specificity log(1 + D/docs(e)).
# Passages participate via MENTIONS: chunk score = summed PPR mass
# of mentioned entities; doc score = max chunk score (the same doc
# aggregation as every other arm).
# ----------------------------------------------------------------------
class PPRIndex:
    def __init__(self, gd: GraphData):
        from scipy import sparse
        self.gd = gd
        nodes = sorted(gd.entity_name)
        self.idx = {eid: i for i, eid in enumerate(nodes)}
        n = len(nodes)
        w: dict[tuple[int, int], float] = defaultdict(float)
        for e in gd.facts:
            i, j = self.idx[e["s_id"]], self.idx[e["o_id"]]
            wt = e["p_true"] * (1.0 if e["tier"] == 1
                                else TIER2_EDGE_DISCOUNT)
            w[(i, j)] += wt
            w[(j, i)] += wt
        rows = [i for i, _ in w]
        cols = [j for _, j in w]
        vals = [v for v in w.values()]
        W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
        out = np.asarray(W.sum(axis=1)).ravel()
        self.dangling = out == 0
        inv = np.where(self.dangling, 0.0, 1.0 / np.where(
            self.dangling, 1.0, out))
        self.P = (W.multiply(inv[:, None])).T.tocsr()  # column form
        D = max(1, gd.n_docs)
        self.specificity = np.array([
            math.log(1.0 + D / max(1, len(gd.entity_docs.get(e, ()))))
            for e in nodes])

    def rank_docs(self, linker_hits: list[dict],
                  triple_sims: np.ndarray | None) -> list[str]:
        gd = self.gd
        s = np.zeros(len(self.idx))
        for h in linker_hits:
            i = self.idx.get(h["entity_id"])
            if i is not None:
                s[i] += (h["score"] / 100.0) * self.specificity[i]
        if triple_sims is not None:
            top = np.argsort(-triple_sims)[:PPR_TOP_TRIPLES]
            for ti in top:
                if triple_sims[ti] <= 0:
                    continue
                e = gd.attestations[int(ti)]
                for eid in (e["s_id"], e["o_id"]):
                    i = self.idx[eid]
                    s[i] += float(triple_sims[ti]) * self.specificity[i]
        if s.sum() <= 0:
            return []
        s /= s.sum()
        r = s.copy()
        for _ in range(100):
            dangling_mass = float(r[self.dangling].sum())
            new = (1 - PPR_ALPHA) * s + PPR_ALPHA * (
                self.P @ r + dangling_mass * s)
            if np.abs(new - r).sum() < 1e-6:
                r = new
                break
            r = new
        ppr = {eid: float(r[i]) for eid, i in self.idx.items()}
        doc_score: dict[str, float] = {}
        for cid, eids in gd.chunk_mentions.items():
            sc = sum(ppr.get(e, 0.0) for e in eids)
            d = gd.chunk_doc[cid]
            if sc > doc_score.get(d, 0.0):
                doc_score[d] = sc
        return [d for d, _ in sorted(doc_score.items(),
                                     key=lambda kv: -kv[1])]


# ----------------------------------------------------------------------
# Chunk arms over GraphStore.search_chunks (the production indexes)
# ----------------------------------------------------------------------
def assemble_pool(gd: GraphData, seed_ids: set, top_docs: set,
                  qtoks: set) -> list[dict]:
    """Evidence pool assembly v2.

    v1 pool: attestations touching a linked seed or a top-ranked
    doc, ordered by (question-token overlap, judged weight). An
    earlier smoke run's 9 correct abstentions traced to deciding
    triples that this pool never reached:

    * conflict questions: the deciding triples hang off the
      parties' CONTRACT entities (one hop beyond the party seeds),
      and a single party's version can fill the pool;
    * multi-hop questions: the answer lives on a LINKED entity's
      facts (2 hops from the seed), outside the 1-hop pool.

    v2 keeps the v1 order as the base and reserves guaranteed
    windows in the pool for two runtime signals (no gold labels —
    both derive from the pool itself):

    * contested-doc-pair interleave (front, <= 16 slots): when
      the seed-touching facts are NOT concentrated in one doc
      (top doc < 60% — a single-doc lookup never triggers this),
      the top-2 seed docs form the contested pair; the relation
      with facts in BOTH docs over the seed frontier (seeds +
      1-hop neighbors) and >= 2 distinct objects is interleaved
      round-robin by doc, so both parties' versions of the
      contested relation are in the pack;
    * 2-hop frontier window (<= 16 slots after the base head):
      facts about 1-hop neighbor entities (the parties' contract
      entities, a linked agreement) — where the deciding slot
      facts for conflict / multi-hop questions actually hang.

    Pool layout: front[:16] + hop2[:16] + base[:28] + the rest,
    so every component is inside the smoke's 60-item window.
    Frontier membership is by ENTITY (a neighbor's fact outranks
    its doc-based base membership — otherwise whole-doc base
    pools bury the frontier, the original v1 failure). Frontier
    ranking uses CONTENT tokens: question tokens minus the seed
    entities' own name tokens, so facts merely restating the
    party's name do not crowd out facts about the asked relation.
    Deterministic: pure function of the graph data and seeds."""
    def w(att):
        return att["p_true"] * (1.0 if att["tier"] == 1
                                else TIER2_EDGE_DISCOUNT)

    seed_touch = [a for a in gd.attestations
                  if a["s_id"] in seed_ids or a["o_id"] in seed_ids]
    neighbors = set()
    for a in seed_touch:
        neighbors.add(a["s_id"])
        neighbors.add(a["o_id"])
    neighbors -= seed_ids

    tok = {id(a): len(qtoks & t)
           for a, t in zip(gd.attestations, gd.att_tokens)}
    seed_name_toks = set()
    for sid in seed_ids:
        seed_name_toks |= {t for t in tokenize(
            gd.entity_name.get(sid, ""))
            if t not in STOP and len(t) > 2}
    ctoks = qtoks - seed_name_toks
    ctok = {id(a): len(ctoks & t)
            for a, t in zip(gd.attestations, gd.att_tokens)}

    # contested pair: seed-touching facts spread over >= 2 docs
    front: list[dict] = []
    doc_mass: dict[str, float] = {}
    for a in seed_touch:
        doc_mass[a["doc_id"]] = doc_mass.get(a["doc_id"], 0.0) + w(a)
    total_mass = sum(doc_mass.values())
    if total_mass > 0:
        ranked_docs = sorted(doc_mass, key=lambda d: -doc_mass[d])
        if doc_mass[ranked_docs[0]] / total_mass < 0.60 \
                and len(ranked_docs) >= 2:
            pair = set(ranked_docs[:2])
            frontier = seed_ids | neighbors
            by_type: dict[str, list[dict]] = {}
            for a in gd.attestations:
                if a["doc_id"] in pair and (
                        a["s_id"] in frontier
                        or a["o_id"] in frontier):
                    by_type.setdefault(a["type"], []).append(a)
            contested, best = None, (-1, -1)
            for t, rows in by_type.items():
                if len({r["doc_id"] for r in rows}) >= 2 \
                        and len({r["o_id"] for r in rows}) >= 2:
                    score = (sum(tok[id(r)] for r in rows), len(rows))
                    if score > best:
                        contested, best = t, score
            if contested is not None:
                rel_rows = by_type[contested]
                rel_rows.sort(key=lambda a: (-ctok[id(a)], -w(a)))
                by_doc: dict[str, list[dict]] = {}
                for a in rel_rows:
                    by_doc.setdefault(a["doc_id"], []).append(a)
                docs_sorted = sorted(by_doc, key=lambda d: -max(
                    w(a) for a in by_doc[d]))
                while any(by_doc.values()):
                    for d in docs_sorted:
                        if by_doc[d]:
                            front.append(by_doc[d].pop(0))
                front = front[:16]
    front_ids = {id(a) for a in front}
    seed_touch_ids = {id(a) for a in seed_touch}
    hop2 = [a for a in gd.attestations
            if id(a) not in front_ids and id(a) not in seed_touch_ids
            and (a["s_id"] in neighbors or a["o_id"] in neighbors)]
    hop2.sort(key=lambda a: (-ctok[id(a)], -w(a)))
    hop2_ids = {id(a) for a in hop2}
    base = [a for a in gd.attestations
            if (a["s_id"] in seed_ids or a["o_id"] in seed_ids
                or a["doc_id"] in top_docs)
            and id(a) not in front_ids and id(a) not in hop2_ids]
    base.sort(key=lambda a: (-tok[id(a)], -w(a)))
    return front + hop2[:16] + base[:28] + base[28:] + hop2[16:]


def dense_chunk_rows(gs, qvec: np.ndarray) -> list[dict]:
    return gs.search_chunks(vector=[float(x) for x in qvec],
                            filters={"batch_id": BATCH_ID},
                            k=CHUNK_FETCH_K)


def text_chunk_rows(gs, question: str) -> list[dict]:
    toks = [t for t in tokenize(question) if len(t) > 2]
    if not toks:
        return []
    return gs.search_chunks(text=" OR ".join(toks),
                            filters={"batch_id": BATCH_ID},
                            k=CHUNK_FETCH_K)


def hybrid_ranking(gs, qvec: np.ndarray,
                   question: str) -> tuple[list[str], list[dict]]:
    """RRF over the dense doc ranking and the Lucene doc ranking
    (the eval's hybrid definition, on the production indexes)."""
    d_rows = dense_chunk_rows(gs, qvec)
    t_rows = text_chunk_rows(gs, question)
    fused = rrf([docs_from_chunk_rows(d_rows),
                 docs_from_chunk_rows(t_rows)])
    return fused, d_rows


# ----------------------------------------------------------------------
# Router (borrow #4, LightRAG dual-level keywords): one Haiku call
# -> entity keywords (low-level) + theme keywords (high-level) +
# route class. Splits are cached and logged per question.
# ----------------------------------------------------------------------
ROUTER_PROMPT = """Split this question for retrieval routing over a contract corpus.
Return ONLY a JSON object, no prose:
{{"entity_keywords": ["..."], "theme_keywords": ["..."], "route": "local"|"global"|"mix"}}
- entity_keywords: specific named parties, contract titles, defined terms the question is about (low-level).
- theme_keywords: concepts/clause types/themes (high-level), e.g. "governing law", "termination", "exclusivity".
- route "local" = about specific linkable entities/contracts; "global" = set, aggregation or thematic across contracts; "mix" = both.
Question: {q}"""


def route_question(question: str, key: str) -> tuple[dict, dict]:
    text, usage = _chat(ROUTER_PROMPT.format(q=question), 300, key)
    try:
        out = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {}
    route = out.get("route")
    if route not in ("local", "global", "mix"):
        route = "mix"
    return {"entity_keywords": out.get("entity_keywords", []),
            "theme_keywords": out.get("theme_keywords", []),
            "route": route}, usage


# ----------------------------------------------------------------------
# Answerer (borrow #7 packing recipes + #13 abstention discipline)
# ----------------------------------------------------------------------
def _norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def verify_span(doc_text: str, span: str) -> bool:
    return bool(span) and _norm_ws(span) in _norm_ws(doc_text)


def pack_evidence(route: str, triples: list[dict],
                  chunk_rows: list[dict]) -> list[dict]:
    """Named packing recipe per route class under the fixed token
    budget: local packs triples first (60/40), global packs chunks
    first (40/60), mix splits evenly. Each item keeps its
    provenance span: a triple's verified quote, or a chunk excerpt
    the answerer must quote verbatim from."""
    shares = {"local": (0.6, 0.4), "global": (0.4, 0.6),
              "mix": (0.5, 0.5)}[route]
    budget = EVIDENCE_TOKEN_BUDGET * 4  # chars (token heuristic /4)
    items: list[dict] = []
    used = 0
    t_budget = budget * shares[0]
    for t in triples:
        span = t.get("quote") or ""
        line = (f"{t['surface_s']} | {t['type']} | "
                f"{t['surface_o']} :: {span}")
        if used + len(line) > t_budget:
            break
        items.append({"kind": "triple", "doc_id": t["doc_id"],
                      "span": span, "text": line,
                      "p_true": t["p_true"], "tier": t["tier"]})
        used += len(line)
    for c in chunk_rows:
        span = c["text"][:600]
        if used + len(span) > budget:
            break
        items.append({"kind": "chunk", "doc_id": c["doc_id"],
                      "span": span, "text": span,
                      "p_true": None, "tier": c.get("tier")})
        used += len(span)
    return items


ANSWER_PROMPT = """Answer the question using ONLY the evidence below. Every factual claim must be supported by a citation: pick the evidence item and copy a verbatim span from it (for triples, the text after ::; for chunks, an exact substring).
Return ONLY a JSON object:
{{"answer": "...", "citations": [{{"evidence": 1, "span": "..."}}], "abstain": false}}
If the evidence does not support an answer, return {{"answer": null, "citations": [], "abstain": true}}.
Evidence:
{evidence}
Question: {q}"""


def answer_question(question: str, route_info: dict,
                    evidence_score: float, linker_hits: list[dict],
                    triple_seed_mass: float, triples: list[dict],
                    chunk_rows: list[dict], texts: dict[str, str],
                    key: str) -> dict:
    """The §7 answer contract: generate from retrieved evidence
    only; citations as (doc_id, span) pairs verified against the
    source documents; abstain — explicitly, with a reason — when
    the ex-ante evidence floor is not met (borrow #13's
    fail_response as a first-class output)."""
    base = {"question": question, "route": route_info["route"],
            "evidence_score": round(float(evidence_score), 4),
            "abstained": False, "answer": None, "citations": [],
            "cost": 0.0}
    if evidence_score < ABSTAIN_FLOOR:
        base.update(abstained=True, reason="evidence_below_floor")
        return base
    if route_info["route"] == "local" and not linker_hits \
            and triple_seed_mass <= 0:
        base.update(abstained=True, reason="no_linkable_evidence")
        return base
    items = pack_evidence(route_info["route"], triples, chunk_rows)
    if not items:
        base.update(abstained=True, reason="empty_evidence")
        return base
    ev_text = "\n".join(
        f"[{i + 1}] (doc {it['doc_id']}, {it['kind']}) {it['text']}"
        for i, it in enumerate(items))
    text, usage = _chat(
        ANSWER_PROMPT.format(evidence=ev_text, q=question), 1200, key)
    base["cost"] = _chat_cost(usage)
    try:
        out = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        out = json.loads(m.group(0)) if m else {}
    if out.get("abstain") or not out.get("answer"):
        base.update(abstained=True, reason="model_abstained")
        return base
    citations = []
    for c in out.get("citations", []):
        try:
            item = items[int(c["evidence"]) - 1]
        except Exception:
            continue
        span = str(c.get("span", ""))
        if verify_span(texts.get(item["doc_id"], ""), span):
            citations.append({"doc_id": item["doc_id"], "span": span,
                              "kind": item["kind"]})
    base["answer"] = out["answer"]
    base["citations"] = citations
    base["citation_verified"] = len(citations) > 0
    if not citations:
        base["reason"] = "no citation resolved to source text"
    return base


# ----------------------------------------------------------------------
# Measurement (§7 contract: the eval arms run THIS code path)
# ----------------------------------------------------------------------
def _load_qa() -> tuple[list[dict], dict[str, str], dict[str, set]]:
    texts, doc_of_file = {}, {}
    for line in open(BATCH_DIR / "texts.jsonl"):
        r = json.loads(line)
        texts[r["doc_id"]] = r["text"]
        doc_of_file[Path(r["source_path"]).name] = r["doc_id"]
    qa = [json.loads(l) for l in open(BATCH_DIR / "qa" / "qa_v1.jsonl")]
    gold = {q["qa_id"]: {doc_of_file[f] for f in q["gold"]["docs"]
                          if f in doc_of_file} for q in qa}
    return qa, texts, gold


def _recall(rank: list[str], gold: set, k: int) -> float:
    return len(gold & set(rank[:k])) / max(1, len(gold))


def _latency_stats(xs: list[float]) -> dict:
    a = np.array(xs) * 1000.0
    return {"mean_ms": round(float(a.mean()), 1),
            "p50_ms": round(float(np.percentile(a, 50)), 1),
            "p95_ms": round(float(np.percentile(a, 95)), 1)}


def _embeddings_cached(name: str, texts_to_embed: list[str],
                       sig: str, embedder) -> tuple[np.ndarray, float]:
    f = EVAL_DIR / name
    if f.exists():
        d = np.load(f, allow_pickle=False)
        if str(d["sig"]) == sig:
            return d["embs"], 0.0
    embs = embedder.embed(texts_to_embed)
    np.savez(f, embs=embs, sig=np.array(sig))
    cost = embedder.usage["cost"]
    return embs, None if cost is None else float(cost)


def measure() -> dict:
    import hashlib

    from .embedder import SeatEmbedder
    from .file_registry import SqliteEntityRegistry
    from .graph_store import GraphStore

    key = _connections.seat("proposer").require_key()
    gs = GraphStore()
    registry = SqliteEntityRegistry(ROOT / "data" / "registry.db")
    qa, texts, gold = _load_qa()

    t0 = time.perf_counter()
    gd = GraphData(gs)
    snapshot_s = time.perf_counter() - t0
    linker = Linker(registry, gs)
    ppr = PPRIndex(gd)
    embedder = SeatEmbedder(EMBED_MODEL)

    q_sig = hashlib.sha256(
        "|".join(q["qa_id"] for q in qa).encode()).hexdigest()
    qembs, q_cost = _embeddings_cached(
        "p3b_qembs.npz", [q["question"] for q in qa], q_sig, embedder)
    t_sig = hashlib.sha256("|".join(
        f"{e['s_id']}|{e['type']}|{e['o_id']}|{e['doc_id']}"
        for e in gd.attestations).encode()).hexdigest()
    triple_texts = [f"{e['surface_s']} {e['type']} {e['surface_o']}"
                    for e in gd.attestations]
    tembs, t_cost = _embeddings_cached(
        "p3b_triple_embs.npz", triple_texts, t_sig, embedder)
    tembs_n = tembs / (np.linalg.norm(tembs, axis=1, keepdims=True)
                       + 1e-9)

    router_cache_f = EVAL_DIR / "p3b_router.json"
    router_cache = json.loads(router_cache_f.read_text()) \
        if router_cache_f.exists() else {}
    router_cost = 0.0
    lat: dict[str, list[float]] = defaultdict(list)
    rows = []
    for qi, q in enumerate(qa):
        question, qid = q["question"], q["qa_id"]
        qvec = qembs[qi]
        if qid not in router_cache:
            t = time.perf_counter()
            info, usage = route_question(question, key)
            lat["router"].append(time.perf_counter() - t)
            router_cost = _connections.add_cost(
                router_cost, _chat_cost(usage))
            router_cache[qid] = {**info, "cost": _chat_cost(usage)}
            router_cache_f.write_text(json.dumps(router_cache, indent=1))
        t = time.perf_counter()
        hits = linker.link(question)
        lat["linker"].append(time.perf_counter() - t)

        t = time.perf_counter()
        d_rows = dense_chunk_rows(gs, qvec)
        lat["dense"].append(time.perf_counter() - t)
        dense_rank = docs_from_chunk_rows(d_rows)
        t = time.perf_counter()
        t_row_list = text_chunk_rows(gs, question)
        hybrid_rank = rrf([dense_rank, docs_from_chunk_rows(t_row_list)])
        lat["hybrid"].append(time.perf_counter() - t
                             + lat["dense"][-1])
        t = time.perf_counter()
        graph_rank = graph_current_ranking(gd, question)
        lat["graph_current"].append(time.perf_counter() - t)
        collapsed_rank = graph_collapsed_ranking(gd, question)
        sims = (tembs_n @ qvec).astype(float)
        t = time.perf_counter()
        ppr_rank = ppr.rank_docs(hits, sims)
        lat["graph_ppr"].append(time.perf_counter() - t)
        arms = {
            "dense": dense_rank,
            "hybrid": hybrid_rank,
            "graph_current": graph_rank,
            "graph_current_collapsed": collapsed_rank,
            "graph_ppr": ppr_rank,
            "graph_hybrid_current": rrf([hybrid_rank, graph_rank]),
            "graph_hybrid_ppr": rrf([hybrid_rank, ppr_rank]),
        }
        g = gold[qid]
        rows.append({
            "qa_id": qid, "class": q["class"], "gold_n": len(g),
            "route": router_cache[qid]["route"],
            "linker_yield": len(hits),
            "recall@20": {a: round(_recall(r, g, 20), 4)
                          for a, r in arms.items()},
            "recall@5": {a: round(_recall(r, g, 5), 4)
                         for a, r in arms.items()},
        })
        print(f"{qid} [{q['class']}] gold={len(g)} " + " ".join(
            f"{a}={rows[-1]['recall@20'][a]:.3f}" for a in arms),
            flush=True)

    arm_names = list(rows[0]["recall@20"])
    overall = {a: round(float(np.mean(
        [r["recall@20"][a] for r in rows])), 4) for a in arm_names}
    overall5 = {a: round(float(np.mean(
        [r["recall@5"][a] for r in rows])), 4) for a in arm_names}
    per_class = {}
    for cls in sorted({r["class"] for r in rows}):
        sub = [r for r in rows if r["class"] == cls]
        per_class[cls] = {a: round(float(np.mean(
            [r["recall@20"][a] for r in sub])), 4) for a in arm_names}
    best_gh = max(("graph_hybrid_current", "graph_hybrid_ppr"),
                  key=lambda a: overall[a])
    overall["graph_hybrid_best"] = overall[best_gh]

    v6 = json.loads((EVAL_DIR / "eval_results_v6.json").read_text())
    v6_overall = {a: round(float(np.mean(
        [r["arms"][a]["recall@20"] for r in v6["rows"]])), 4)
        for a in ("dense", "hybrid", "graph", "graph_hybrid")}

    out = {
        "n_questions": len(rows),
        "attestation_join": gd.attestation_join,
        "snapshot_load_s": round(snapshot_s, 2),
        "overall_recall@20": overall, "overall_recall@5": overall5,
        "per_class_recall@20": per_class,
        "v6_reference_overall": v6_overall,
        "latency": {a: _latency_stats(v) for a, v in lat.items()},
        "spend": {"question_embeddings":
                      None if q_cost is None else round(q_cost, 4),
                  "triple_embeddings":
                      None if t_cost is None else round(t_cost, 4),
                  "router":
                      None if router_cost is None
                      else round(router_cost, 4),
                  "embedder_usage_total": embedder.usage},
        "rows": rows,
    }
    (EVAL_DIR / "p3b_ab.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in (
        "overall_recall@20", "v6_reference_overall", "latency",
        "spend")}, indent=1))
    return out


def smoke() -> dict:
    from .embedder import SeatEmbedder
    from .file_registry import SqliteEntityRegistry
    from .graph_store import GraphStore

    key = _connections.seat("proposer").require_key()
    gs = GraphStore()
    registry = SqliteEntityRegistry(ROOT / "data" / "registry.db")
    qa, texts, gold = _load_qa()
    gd = GraphData(gs)
    linker = Linker(registry, gs)
    ppr = PPRIndex(gd)
    embedder = SeatEmbedder(EMBED_MODEL)

    picked: list[dict] = []
    for cls in sorted({q["class"] for q in qa}):
        picked += [q for q in qa if q["class"] == cls][:3]
    router_cache_f = EVAL_DIR / "p3b_router.json"
    router_cache = json.loads(router_cache_f.read_text()) \
        if router_cache_f.exists() else {}
    tembs_f = EVAL_DIR / "p3b_triple_embs.npz"
    tembs_n = None
    if tembs_f.exists():
        tembs = np.load(tembs_f)["embs"]
        tembs_n = tembs / (np.linalg.norm(tembs, axis=1, keepdims=True)
                           + 1e-9)
    out_rows = []
    total_cost = 0.0
    for q in picked:
        question, qid = q["question"], q["qa_id"]
        if qid not in router_cache:
            info, usage = route_question(question, key)
            router_cache[qid] = {**info, "cost": _chat_cost(usage)}
            router_cache_f.write_text(json.dumps(router_cache, indent=1))
        info = router_cache[qid]
        hits = linker.link(question)
        qvec = embedder.embed([question])[0]
        d_rows = dense_chunk_rows(gs, qvec)
        evidence_score = max(
            (r.get("vector_score", 0.0) for r in d_rows), default=0.0)
        t_rows = text_chunk_rows(gs, question)
        seen_chunks = set()
        chunk_rows = []
        for r in d_rows[:6] + t_rows[:6]:
            if r["chunk_id"] not in seen_chunks:
                seen_chunks.add(r["chunk_id"])
                chunk_rows.append(r)
        sims = (tembs_n @ qvec).astype(float) \
            if tembs_n is not None else None
        ppr_rank = ppr.rank_docs(hits, sims)
        seed_mass = float(sims[sims.argsort()[-PPR_TOP_TRIPLES:]].clip(
            min=0).sum()) if sims is not None else 0.0
        seed_ids = {h["entity_id"] for h in hits}
        top_docs = set(ppr_rank[:10]) | set(rrf(
            [docs_from_chunk_rows(d_rows),
             docs_from_chunk_rows(t_rows)])[:10])
        # evidence pool: assembly v2 (assemble_pool) — v1 seed /
        # top-doc pool plus contested-relation interleave and the
        # 2-hop frontier (see assemble_pool's docstring)
        qtoks = {t for t in tokenize(question)
                 if t not in STOP and len(t) > 2}
        pool = assemble_pool(gd, seed_ids, top_docs, qtoks)
        res = answer_question(
            question, info, evidence_score, hits, seed_mass,
            pool[:60], chunk_rows, texts, key)
        total_cost = _connections.add_cost(total_cost, res["cost"])
        res.update({"qa_id": qid, "class": q["class"],
                    "linker_yield": len(hits),
                    "citation_count": len(res["citations"])})
        out_rows.append(res)
        print(f"{qid} [{q['class']}] route={info['route']} "
              f"abstained={res['abstained']} "
              f"citations={len(res['citations'])} "
              f"verified={res.get('citation_verified')}", flush=True)
    out = {"n": len(out_rows),
           "abstained": sum(r["abstained"] for r in out_rows),
           "answered_with_verified_citation": sum(
               1 for r in out_rows if not r["abstained"]
               and r.get("citation_verified")),
           "answerer_cost": None if total_cost is None
           else round(total_cost, 4), "rows": out_rows}
    (EVAL_DIR / "p3b_smoke.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="pipeline.query_stack")
    ap.add_argument("cmd", choices=("project", "measure", "smoke"))
    args = ap.parse_args()
    if args.cmd == "project":
        print(json.dumps(project_serving(), indent=1))
    elif args.cmd == "measure":
        measure()
    else:
        print(json.dumps(smoke(), indent=1,
                         default=str)[:2000])


if __name__ == "__main__":
    main()
