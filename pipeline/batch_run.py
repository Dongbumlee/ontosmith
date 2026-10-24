"""Batch runner: executes the 8-stage pipeline for a Batch.

Stages: INGEST → NER → CANDIDATES → GATES → ROUTE → WRITE → INDEX → VERIFY
Each stage checkpoints; a crash resumes from the last completed stage.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .batch import (
    Batch,
    BatchStatus,
    advance,
    batch_dir,
    doc_id_for_text,
    git_code_version,
    mark_failed,
    new_run_id,
    save_batch,
    save_state,
    write_manifest,
)
from .heartbeat import beat as _heartbeat_beat


def _log(batch_id: str, msg: str) -> None:
    print(f"[batch {batch_id}] {msg}", flush=True)


def _beat(b: Batch, root: Path, stage: str,
          progress: int | None = None, total: int | None = None) -> None:
    """Liveness heartbeat (§3): rewritten at stage transitions and at
    CANDIDATES/GATES checkpoints. Never raises (see heartbeat.py)."""
    _heartbeat_beat(batch_dir(root, b.batch_id), b.batch_id, b.run_id,
                    stage, progress, total)


def _conn_add_cost(total, add):
    # None-aware spend accumulation (connections.add_cost).
    from . import connections as _conn
    return _conn.add_cost(total, add)


def _round_cost(usd):
    # Round a possibly-unmetered (None) spend figure.
    return None if usd is None else round(usd, 4)


def _fmt_cost(usd) -> str:
    # Log-line spend: unmetered reads as such, not $0.0000.
    return "unmetered" if usd is None else f"${usd:.4f}"


def _record_cost(b: Batch, stage: str, usd: float | None) -> None:
    """Cost ledger (§3): per-stage API spend in batch metrics.

    b.metrics["cost_ledger"] maps stage -> USD (assignment semantics:
    a resumed stage re-records its recomputed total, so the ledger
    never double-counts) plus "total_usd", "cap_usd" and
    "cost_basis" (the bindings' aggregate basis —
    provider-reported / estimated / not-provided / mixed).
    Enforces config.max_cost_usd as a hard cap: recording a stage
    that pushes the cumulative total over the cap raises, which
    aborts the run (run_batch marks the batch FAILED with this
    error in state).

    `usd` may be None — the stage's spend is unmetered (a
    cost-less binding). The ledger records None (unknown),
    never a folded 0.0. While any recorded stage is
    unmetered the run's total is likewise unknown:
    "total_usd" is None and the metered stages' sum rides
    in "metered_usd" (a bare numeric total beside
    a null stage told JSON consumers the run cost that
    much, or nothing, when the truth is "partly unknown").
    With a cap set, an unmetered stage refuses by name
    instead: a cap cannot guard spend it cannot meter
    (create/run preflights refuse this combination
    earlier; this is the same rule at the point of record).
    """
    from . import connections as _conn
    cap = b.config.max_cost_usd
    if usd is None and cap is not None:
        raise _conn.BindingCostError(
            f"stage {stage} recorded unmetered spend (the "
            f"binding reports no cost) while max_cost_usd "
            f"${cap:.2f} is set; a USD cap cannot guard spend "
            f"it cannot meter — bind a cost-reporting route, "
            f"declare per-seat rates, or remove the cap "
            f"(--no-cost-cap)")
    ledger = dict(b.metrics.get("cost_ledger") or {})
    ledger[stage] = None if usd is None else round(float(usd), 4)
    stage_values = [v for k, v in ledger.items()
                    if k not in ("total_usd", "metered_usd",
                                 "cap_usd", "cost_basis")]
    metered = round(sum(v for v in stage_values
                        if isinstance(v, (int, float))), 4)
    ledger["metered_usd"] = metered
    total = None if any(v is None for v in stage_values) \
        else metered
    ledger["total_usd"] = total
    ledger["cap_usd"] = cap
    ledger["cost_basis"] = _conn.resolve_bindings().cost_basis()
    b.metrics["cost_ledger"] = ledger
    if cap is not None and total > cap:
        raise RuntimeError(
            f"cost cap exceeded: ledger total ${total:.4f} > "
            f"max_cost_usd ${cap:.2f} "
            f"(stage {stage} recorded ${usd:.4f})")


def _check_cost_cap(b: Batch) -> None:
    """Pre-stage cap guard: refuse to start a stage when the ledger
    already sits at/over the hard cap. No cap (None) never
    refuses."""
    cap = b.config.max_cost_usd
    if cap is None:
        return
    ledger = b.metrics.get("cost_ledger") or {}
    total = ledger.get("total_usd") or 0.0
    if total >= cap:
        raise RuntimeError(
            f"cost cap exceeded: ledger total ${total:.4f} >= "
            f"max_cost_usd ${cap:.2f} before stage start")


def _load_setup_config(b: Batch, root: Path) -> dict | None:
    """Stage-0 setup_config for the batch's profile family.

    Consumed by CANDIDATES (proposer categories, role-word
    rules, guidance, endpoint types), ROUTE
    (C1 whitelist) and WRITE (relation map, merge
    prohibitions). The registered consumer set lives in
    pipeline/profile_config.py; batch creation fails closed on
    any key outside it. None when the batch runs without a
    profile; stages then use their no-profile defaults.

    Sources, in priority order:
    1. the batch directory's ``setup_config.resolved.json`` —
       the resolver's output frozen at creation (core
       baseline + the binding pack); a batch executes exactly
       the configuration its creation resolved;
    2. the legacy ``profiles/<family>/setup_config.json``;
    3. the family's profile file in the pack layout
       (``profile_setup_config_path``).
    """
    if not b.profile_family:
        return None
    from .profile_config import profile_setup_config_path
    resolved = batch_dir(root, b.batch_id) / "setup_config.resolved.json"
    legacy = root / "profiles" / b.profile_family / "setup_config.json"
    candidates = [resolved, legacy]
    pack_sc = profile_setup_config_path(root, b.profile_family)
    if pack_sc is not None:
        candidates.append(pack_sc)
    p = next((c for c in candidates if c.exists()), None)
    # The batch DECLARES a profile family, so a missing
    # or unreadable setup_config is a hard failure, not a silent
    # no-profile fallback — running under defaults while batch.json
    # names a profile makes the run's semantics differ from its
    # record with no trace. Fail loud; the stage runner records the
    # error on the batch (FAILED + error) like any other stage fault.
    if p is None:
        raise FileNotFoundError(
            f"batch declares profile_family={b.profile_family!r} but "
            f"no setup_config exists at any of "
            f"{[str(c) for c in candidates]}; refusing to run with "
            "no-profile defaults under a named profile")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise RuntimeError(
            f"setup_config for profile_family={b.profile_family!r} "
            f"at {p} is unreadable/corrupt: {e}; refusing to run "
            "with no-profile defaults under a named profile") from e


def stage_ingest(b: Batch, root: Path, doc_texts: dict[str, str],
                 stages: list[str]) -> None:
    """INGEST: texts → texts.jsonl with content-hash doc_ids. Dedupe."""
    _log(b.batch_id, f"INGEST: {len(doc_texts)} input docs")
    d = batch_dir(root, b.batch_id)
    seen: dict[str, dict] = {}
    sources = []
    with open(d / "texts.jsonl", "w") as f:
        for path, text in doc_texts.items():
            doc_id = doc_id_for_text(text)
            if doc_id in seen:
                continue
            seen[doc_id] = {"path": path}
            f.write(json.dumps({
                "doc_id": doc_id, "source_path": path,
                "text": text, "char_count": len(text)}) + "\n")
            sources.append({"doc_id": doc_id, "source_path": path,
                            "char_count": len(text)})
    b.doc_ids = sorted(seen.keys())
    b.metrics["ingest"] = {"docs_in": len(doc_texts),
                           "docs_unique": len(seen),
                           "docs_deduped": len(doc_texts) - len(seen)}
    _record_cost(b, "INGEST", 0.0)  # local only, no API spend
    write_manifest(b, sources, root)
    save_batch(b, root)
    advance(b, root, [*stages, "INGEST"])
    _log(b.batch_id, f"INGEST done: {len(seen)} unique docs")


def _ner_chunks(text: str) -> list[str]:
    """Chunk selection for NER: 2,000-char slices, at most 10 per doc.

    Dispersed sampling (2026-10-12): the 10-chunk budget
    is spread evenly across the whole document (first and last included)
    instead of head-only [:10], which covered only ~14.4% of corpus
    chars, all from document heads.
    """
    all_chunks = [text[j:j + 2000] for j in range(0, len(text), 2000)]
    if len(all_chunks) > 10:
        pick = sorted({round(k * (len(all_chunks) - 1) / 9)
                       for k in range(10)})
        return [all_chunks[i] for i in pick]
    return all_chunks


def _ner_env_fixes() -> None:
    """Env fixes needed before GLiNER loads (also in worker processes)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # httpx.Client() parses no_proxy/NO_PROXY at construction; bracketed
    # IPv6 entries like [::1] crash its URLPattern parser. Strip them.
    for var in ("no_proxy", "NO_PROXY"):
        val = os.environ.get(var, "")
        if "[" in val:
            os.environ[var] = ",".join(
                p for p in val.split(",") if "[" not in p)


def _predict_doc_entities(model, chunks: list[str], labels: list[str],
                          stats: dict | None = None):
    """Entities for one doc: batched prediction, per-chunk fallback.

    Dedupes on (text.lower(), label), preserving first-seen order —
    identical semantics to the original sequential loop.

    Failure accounting: the batched-call
    failure and every swallowed per-chunk failure increment
    ``stats`` when one is passed — a chunk the model choked on
    must not be indistinguishable from a chunk with no entities.
    """
    def _bump(field: str) -> None:
        if stats is not None:
            stats[field] = stats.get(field, 0) + 1

    ents = []
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_chunk = model.batch_predict_entities(
                chunks, labels, threshold=0.5)
        for chunk_ents in per_chunk:
            for e in chunk_ents:
                ents.append({"text": e["text"], "label": e["label"],
                             "score": round(e["score"], 3)})
    except Exception:
        _bump("ner_batch_predict_failures")
        for ch in chunks:
            try:
                for e in model.predict_entities(ch, labels, threshold=0.5):
                    ents.append({"text": e["text"], "label": e["label"],
                                 "score": round(e["score"], 3)})
            except Exception:
                _bump("ner_chunk_failures")
    seen, uniq = set(), []
    for e in ents:
        k = (e["text"].lower(), e["label"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


_NER_WORKER_MODEL = None
_NER_WORKER_LABELS = ["person", "organization", "location", "date", "event"]


def _ner_worker_init(model_name: str) -> None:
    global _NER_WORKER_MODEL
    _ner_env_fixes()
    from gliner import GLiNER
    _NER_WORKER_MODEL = GLiNER.from_pretrained(model_name)


def _ner_worker_doc(task):
    """One doc in a worker process. Returns (doc_id, entities,
    failed, stats): a doc whose prediction raised is reported as
    failed with an empty entity list (the catch-all used
    to return a bare [] indistinguishable from a doc that simply
    has no entities, and the stage recorded neither)."""
    doc_id, chunks = task
    stats: dict = {}
    try:
        return doc_id, _predict_doc_entities(
            _NER_WORKER_MODEL, chunks, _NER_WORKER_LABELS, stats), \
            False, stats
    except Exception:
        return doc_id, [], True, stats


def stage_ner(b: Batch, root: Path, stages: list[str]) -> None:
    """NER: GLiNER over chunked text → ner.jsonl.

    Parallel across documents (JEV_NER_WORKERS worker processes, each
    with its own GLiNER instance; default 2). The main process is the
    single writer — workers return entities, appends stay ordered by
    completion and resume semantics are unchanged. On many-core hosts
    raise JEV_NER_WORKERS; each worker costs ~1 model of RAM.
    """
    _ner_env_fixes()
    labels = list(_NER_WORKER_LABELS)
    d = batch_dir(root, b.batch_id)
    texts = {}
    with open(d / "texts.jsonl") as f:
        for line in f:
            r = json.loads(line)
            texts[r["doc_id"]] = r["text"]
    if (b.config.candidate_source or "pipeline").lower() == "llm":
        # GLiNER removed from the LLM path (ablation
        # verdict): the proposer's guards attest by deterministic
        # text containment and never consult NER mentions; GLiNER's
        # only llm-path role was the document list, which
        # texts.jsonl already provides. Emit one empty-entity row
        # per doc so the ner.jsonl contract and resume semantics
        # are unchanged, and never load the model. Pipeline/both
        # sources are unaffected (they consume the mentions).
        ner_path = d / "ner.jsonl"
        have = set()
        if ner_path.exists():
            for line in ner_path.read_text().splitlines():
                try:
                    have.add(json.loads(line)["doc_id"])
                except Exception:
                    break
        with open(ner_path, "a") as out:
            for doc_id in texts:
                if doc_id not in have:
                    out.write(json.dumps({"doc_id": doc_id,
                                          "entities": []}) + "\n")
        b.metrics["ner"] = {"entities": 0, "entities_per_doc": 0.0,
                            "seconds": 0.0, "workers": 0,
                            "gliner_skipped_llm_path": True}
        _record_cost(b, "NER", 0.0)
        save_batch(b, root)
        advance(b, root, [*stages, "NER"])
        _log(b.batch_id, "NER: GLiNER skipped (candidate_source=llm; "
                         "removed from the LLM path)")
        return
    total_ents = 0
    t0 = time.perf_counter()
    # Resume support: a killed NER stage previously lost all
    # completed docs. Keep valid lines, drop a possibly truncated
    # tail line, skip finished doc_ids.
    ner_path = d / "ner.jsonl"
    done_ids = set()
    if ner_path.exists():
        good = []
        for line in ner_path.read_text().splitlines():
            try:
                rec = json.loads(line)
                done_ids.add(rec["doc_id"])
                good.append(line)
                total_ents += len(rec.get("entities", []))
            except Exception:
                break
        ner_path.write_text("".join(l + "\n" for l in good))
        if done_ids:
            _log(b.batch_id, f"NER: resuming, {len(done_ids)} docs already done")
    todo = [(doc_id, _ner_chunks(text)) for doc_id, text in texts.items()
            if doc_id not in done_ids]
    workers = max(1, int(os.environ.get("JEV_NER_WORKERS", "2")))
    done_count = len(done_ids)

    def _emit(out, doc_id, uniq):
        nonlocal total_ents, done_count
        total_ents += len(uniq)
        done_count += 1
        out.write(json.dumps({"doc_id": doc_id, "entities": uniq}) + "\n")
        out.flush()
        if done_count % 10 == 0:
            _log(b.batch_id, f"NER: {done_count}/{len(texts)} docs")
            _beat(b, root, "NER", done_count, len(texts))

    ner_stats: dict = {}
    ner_failed_docs = 0
    with open(ner_path, "a") as out:
        if workers <= 1 or len(todo) <= 1:
            _log(b.batch_id, "NER: loading GLiNER")
            from gliner import GLiNER
            model = GLiNER.from_pretrained(b.config.ner_model)
            for doc_id, chunks in todo:
                _emit(out, doc_id, _predict_doc_entities(
                    model, chunks, labels, ner_stats))
        else:
            _log(b.batch_id, f"NER: {workers} worker processes")
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                    max_workers=workers, mp_context=ctx,
                    initializer=_ner_worker_init,
                    initargs=(b.config.ner_model,)) as ex:
                futs = {ex.submit(_ner_worker_doc, t): t[0] for t in todo}
                for fut in as_completed(futs):
                    doc_id, uniq, failed, wstats = fut.result()
                    for k, v in wstats.items():
                        ner_stats[k] = ner_stats.get(k, 0) + v
                    if failed:
                        ner_failed_docs += 1
                    _emit(out, doc_id, uniq)
    dt = time.perf_counter() - t0
    b.metrics["ner"] = {"entities": total_ents,
                        "entities_per_doc": round(total_ents / max(len(texts), 1), 1),
                        "seconds": round(dt, 1),
                        "workers": workers,
                        "failed_docs": ner_failed_docs,
                        "chunk_failures": ner_stats.get(
                            "ner_chunk_failures", 0),
                        "batch_predict_failures": ner_stats.get(
                            "ner_batch_predict_failures", 0)}
    _record_cost(b, "NER", 0.0)  # local GLiNER inference, no API spend
    save_batch(b, root)
    advance(b, root, [*stages, "NER"])
    _log(b.batch_id, f"NER done: {total_ents} entities")

def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter. Good enough for candidate context."""
    import re
    # Split on sentence boundaries, keep the delimiter
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


# Common nouns that NER sometimes mislabels as entities.
# These are not named entities and should not become graph nodes.
_STOP_ENTITIES = frozenset([
    "man", "woman", "person", "people", "child", "boy", "girl",
    "president", "director", "agent", "officer", "member",
    "government", "committee", "party", "group",
    "city", "country", "state", "nation",
    "year", "month", "day", "time",
    "p", "a", "i",  # single letters from OCR
])


def _is_valid_entity(text: str) -> bool:
    """Filter out OCR garbage and common nouns before candidate generation.

    Design (2026-10-11): entity quality gate at CANDIDATES stage.
    The pipeline doesn't clean input (§10.3), but it shouldn't waste gate
    judgments on obvious non-entities either.
    """
    import re
    t = text.strip()
    # Too short (but allow 2-char acronyms like UN, US, UK)
    if len(t) < 2:
        return False
    if len(t) == 2 and not (t.isupper() or t.lower() in ("us", "uk", "un", "eu")):
        return False
    # Common noun mislabeled as entity
    if t.lower() in _STOP_ENTITIES:
        return False
    # OCR garbage: excessive internal whitespace
    if re.search(r'\s{3,}', t):
        return False
    # OCR garbage: mostly non-alphabetic
    alpha = sum(1 for c in t if c.isalpha())
    if alpha < len(t) * 0.5:
        return False
    return True


def stage_candidates(b: Batch, root: Path, stages: list[str]) -> None:
    """CANDIDATES: RER-based triple extraction → candidates.

    Design evolution (2026-10-11):
    - v1: document-level pairing (mechanical co-occurrence)
    - v2: sentence-level pairing (contextual, but relation=associated_with)
    - v3: RER via spaCy — extract actual verb as relation.
    - v4 (current): Hybrid RER — spaCy → LLM (DeepSeek) → v2 pairing.
      LLM fallback handles OCR-noisy text where spaCy fails.

    The verb becomes the relation. Jev still validates.

    Candidate source (config.candidate_source):
    - "pipeline": the RER hybrid above, unchanged.
    - "llm": the LLM proposer only (pipeline/proposer.py) — free
      triples plus clause slots where the profile declares
      categories; deterministic span/attestation guards at
      proposal time, Jev gates judge downstream as usual.
    - "both": union of the two, deduped on the normalized key.
    """
    source = (b.config.candidate_source or "pipeline").lower()
    if source not in ("pipeline", "llm", "both"):
        raise ValueError(f"unknown candidate_source: {source!r}")
    use_pipeline = source in ("pipeline", "both")
    use_llm = source in ("llm", "both")
    setup_cfg = _load_setup_config(b, root) if use_llm else None
    slot_categories = list((setup_cfg or {}).get(
        "label_categories") or [])
    # The role-word guard vocabulary is pack data,
    # resolved by profile_config — frozen into the resolved
    # config at batch creation, or read from the binding
    # pack's role_words artefact. None = the profile declares
    # no role-word rules (guard off); a declaration with no
    # resolvable vocabulary refuses here (fail closed).
    from .profile_config import effective_role_words
    role_words = effective_role_words(root, setup_cfg)
    # profile wiring: the guidance
    # rendered from the profile's setup_config is threaded into the
    # proposer prompt, and proposed endpoint types are validated
    # against the profile's endpoint-type vocabulary. A profile
    # that configures no guidance yields "" here, and the prompt
    # construction is then byte-identical to the pre-channel one.
    from .profile_config import (
        derive_guidance,
        endpoint_type_names,
        pair_judge_generic_relation,
        proposer_framing,
        slot_framing,
        slot_rules,
    )
    guidance = derive_guidance(setup_cfg) if use_llm else ""
    endpoint_types = endpoint_type_names(setup_cfg) if use_llm \
        else set()
    # Domain framing: the family's declared framing
    # for the proposer / pair-judge base templates, threaded as
    # data; None (profile declares none) selects the builders'
    # domain-neutral core defaults.
    triple_framing = proposer_framing(setup_cfg) if use_llm \
        else None
    clause_framing = slot_framing(setup_cfg) if use_llm else None
    clause_rules = slot_rules(setup_cfg) if use_llm else None
    pair_generic_relation = pair_judge_generic_relation(
        setup_cfg) if use_llm else None
    if use_llm:
        _log(b.batch_id,
             f"CANDIDATES: LLM proposer (model={b.config.llm_model}, "
             f"slot categories={len(slot_categories)}, "
             f"role-word guard={'on' if role_words else 'off'})")
        if guidance:
            _log(b.batch_id,
                 f"CANDIDATES: profile guidance active "
                 f"({len(guidance)} chars); endpoint types "
                 f"validated against {sorted(endpoint_types)}")
    nlp = None
    if use_pipeline:
        import spacy

        from .rer import (
            extract_triples,
            extract_triples_constrained,
            extract_triples_llm_with_usage,
        )
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = None
            _log(b.batch_id,
                 "CANDIDATES: spaCy model not found, using fallback")

    d = batch_dir(root, b.batch_id)
    texts = {}
    source_paths = {}
    with open(d / "texts.jsonl") as f:
        for line in f:
            r = json.loads(line)
            texts[r["doc_id"]] = r["text"]
            source_paths[r["doc_id"]] = r.get("source_path", "")
    # Pair-enumeration branch: with the profile's
    # pair_enumeration switch on, the proposer phase below
    # replaces the free generative proposer with the roster pair
    # census (pipeline/pair_proposer.py). The roster is a frozen
    # batch-dir input: roster_v2.jsonl when the profile declares
    # roster_pass (the mention-first pass product),
    # roster.jsonl otherwise. Switch absent/off -> this
    # block is inert and candidate construction is byte-identical
    # to the pre-branch behavior (pinned by the wiring tests).
    pair_enum = bool(use_llm and setup_cfg
                     and setup_cfg.get("pair_enumeration"))
    pair_window = (setup_cfg or {}).get("pair_window") or "sentence"
    # Pair guard relaxations: profile-selected,
    # absent = the frozen legacy behavior (pinned by
    # tests/test_emission_guards.py). theme attestation for
    # pair items uses the pair sentence the endpoints were
    # posed from; the quote check additionally accepts
    # punctuation-spacing-equivalent quotes.
    pair_abbrev_guard = bool((setup_cfg or {}).get(
        "pair_abbrev_guard"))
    pair_theme_attest = bool((setup_cfg or {}).get(
        "pair_theme_attest_sentence"))
    quote_punct_insensitive = bool((setup_cfg or {}).get(
        "quote_punct_insensitive"))
    # theme attestation scope for free proposals —
    # profile/pack-selected, absent = the quote-only test.
    theme_attest_chunk = bool((setup_cfg or {}).get(
        "theme_attest_chunk"))
    # Pair-judge variant: profile-selected;
    # absent = the default judge (sentence x standard).
    pair_judge_context = (setup_cfg or {}).get(
        "pair_judge_context") or "sentence"
    pair_judge_instruction = (setup_cfg or {}).get(
        "pair_judge_instruction") or "standard"
    # JF-F1 direction slot on the production path:
    # profile-selected via the registered pair_judge_format key.
    # Absent = the legacy answer format — the proposer is then
    # called with jf1=False and a None symmetric set, exactly the
    # pre-existing call, and prompts / parsing / emission / verdict
    # records are byte-identical (pinned by the wiring tests). The symmetric set
    # is DATA: read from the pinned pack artefact the profile
    # names (pair_judge_symmetric_source), never a core literal;
    # any failure to resolve it fails the stage closed.
    pair_judge_format = (setup_cfg or {}).get("pair_judge_format")
    if pair_judge_format is not None and pair_judge_format != "jf1":
        raise ValueError(
            f"pair_judge_format {pair_judge_format!r} is not a "
            "registered format (fail closed; registered: 'jf1', "
            "or omit the key for the legacy format)")
    jf1_on = bool(pair_enum and pair_judge_format == "jf1")
    symmetric_set: set[str] | None = None
    symmetric_src: dict | None = None
    if jf1_on:
        src = (setup_cfg or {}).get("pair_judge_symmetric_source")
        if not (isinstance(src, str) and src.strip()):
            raise ValueError(
                "pair_judge_format 'jf1' requires "
                "'pair_judge_symmetric_source' (fail closed)")
        spath = Path(src)
        if not spath.is_absolute():
            # Pack-declared artefact paths are workspace-relative:
            # they resolve against the batch state root (the tree
            # beside the code in a checkout; $ONTOSMITH_ROOT / the
            # working directory for an installed copy), never
            # against the package's own tree.
            spath = root / spath
        try:
            artefact = json.loads(spath.read_text())
            sym = artefact["symmetric_relations"]
            if not (isinstance(sym, list) and sym and all(
                    isinstance(s, str) and s for s in sym)):
                raise ValueError(
                    "symmetric_relations is not a non-empty "
                    "string list")
        except Exception as e:
            raise ValueError(
                f"JF-F1 symmetric-set artefact {spath} is "
                f"unreadable or malformed: {e} (fail closed)"
            ) from e
        symmetric_set = set(sym)
        symmetric_src = {
            "path": src,
            "sha256": hashlib.sha256(spath.read_bytes()).hexdigest(),
            "n_symmetric": len(symmetric_set)}
    relation_inventory: set[str] = set()
    roster_by_doc: dict[str, list[dict]] = {}
    roster_name = ""
    if pair_enum:
        sf = (setup_cfg or {}).get("schema_front") or {}
        relation_inventory = {
            str(r["type"]) for r in sf.get("relation_inventory") or []
            if isinstance(r, dict) and r.get("type")} or {
            str(x) for x in (setup_cfg or {}).get("relation_types")
            or []}
        roster_name = "roster_v2.jsonl" if (setup_cfg or {}).get(
            "roster_pass") else "roster.jsonl"
        roster_path = d / roster_name
        if not roster_path.exists():
            raise RuntimeError(
                f"pair_enumeration is on but the batch roster "
                f"{roster_name} is missing from {d} — materialize "
                f"the frozen roster into the batch dir first")
        for line in open(roster_path):
            if line.strip():
                rr = json.loads(line)
                roster_by_doc[rr["doc_id"]] = rr["mentions"]
        _log(b.batch_id,
             f"CANDIDATES: pair enumeration ON (window="
             f"{pair_window}, roster={roster_name}, judge="
             f"{pair_judge_context}x{pair_judge_instruction}: "
             f"{len(roster_by_doc)} docs, "
             f"{sum(len(v) for v in roster_by_doc.values())} "
             f"mentions, inventory={len(relation_inventory)} "
             f"relations)")
        if jf1_on:
            _log(b.batch_id,
                 f"CANDIDATES: JF-F1 direction slot ON (symmetric "
                 f"set: {symmetric_src['n_symmetric']} relations "
                 f"from {symmetric_src['path']}, sha256 "
                 f"{symmetric_src['sha256'][:12]}…)")
    # NER mentions feed only the pipeline pairing phases; the LLM
    # proposer never consults them. For llm-only
    # batches the document list comes from texts.jsonl and
    # ner.jsonl is not read at all.
    ner_data = {}
    if use_pipeline:
        with open(d / "ner.jsonl") as f:
            for line in f:
                r = json.loads(line)
                ner_data[r["doc_id"]] = r["entities"]

    candidates = []
    blocked_all = []  # Arm-B C3-blocked records (killed-set audit sidecar)
    # Verdict sidecar:
    # every pair verdict the pair stage produces, persisted
    # pre-guard so the verdict -> blocked -> deduped -> kept
    # accounting is reproducible from artifacts alone.
    pair_verdicts: list[dict] = []
    pair_verdicts_consistent = True
    llm_cost = 0.0  # LLM-fallback spend, summed across docs
    # Proposer spend/stats. Persisted in the checkpoint so a
    # resumed run keeps the true totals (unlike the fallback above).
    proposer_cost = 0.0
    proposer_stats = {"proposed": 0, "calls": 0, "errors": 0,
                      "slots_proposed": 0, "typed_kept": 0,
                      "types_dropped_unknown": 0}
    pair_stats = {"enumerated": 0, "answered": 0, "calls": 0,
                  "outcomes": {}, "alias_emitted": 0}
    # Silent-path counters: the stage's
    # swallowed failures are counted and stamped into the stage
    # metrics — per-doc candidate-cap hits, Phase-2 worker
    # exceptions, RER-LLM fallback failures (delta of rer's call
    # stats over this stage), and a corrupted-checkpoint reset.
    cand_counters = {"cap_docs": 0, "cap_dropped": 0,
                     "llm_worker_errors": 0}
    ckpt_reset: dict | None = None
    from . import rer as _rer_mod
    rer_stats0 = _rer_mod.llm_call_stats()
    rer_env = {"JEV_RER_V2": os.environ.get("JEV_RER_V2") == "1",
               "JEV_RER_LLM": os.environ.get("JEV_RER_LLM") == "1"}
    # Checkpoint for resume: track processed doc_ids
    ckpt_path = d / "candidates.ckpt.json"
    done_docs = set()
    if ckpt_path.exists():
        try:
            ckpt = json.loads(ckpt_path.read_text())
            done_docs = set(ckpt.get("done_docs", []))
            _ckpt_cost = ckpt.get("proposer_cost_usd", 0.0)
            proposer_cost = (None if _ckpt_cost is None
                             else float(_ckpt_cost))
            for k in proposer_stats:
                proposer_stats[k] = int(ckpt.get("proposer_stats", {})
                                        .get(k, 0))
            for k in pair_stats:
                pair_stats[k] = ckpt.get("pair_stats", {}).get(
                    k, pair_stats[k])
            # Load existing candidates
            if (d / "candidates.jsonl").exists():
                with open(d / "candidates.jsonl") as f:
                    for line in f:
                        candidates.append(json.loads(line))
                _log(b.batch_id, f"CANDIDATES: resumed with {len(done_docs)} docs done, "
                    f"{len(candidates)} candidates")
            if (d / "candidates_blocked.jsonl").exists():
                with open(d / "candidates_blocked.jsonl") as f:
                    for line in f:
                        blocked_all.append(json.loads(line))
            if pair_enum and (d / "pair_verdicts.jsonl").exists():
                with open(d / "pair_verdicts.jsonl") as f:
                    for line in f:
                        if line.strip():
                            pair_verdicts.append(json.loads(line))
            elif pair_enum and pair_stats.get("outcomes"):
                # Resumed from a pre-sidecar checkpoint: the metrics
                # carry verdicts the sidecar never saw. The sidecar
                # then covers only docs judged by the wired code,
                # and the outcome-count assert below is skipped.
                pair_verdicts_consistent = False
        except Exception as e:
            # A checkpoint that cannot be read resets the stage's
            # resume state (done-docs, proposer/pair stats, loaded
            # candidates). That reset used to be silent:
            # log it and stamp it into the metrics.
            ckpt_reset = {"error": f"{type(e).__name__}: {e}"}
            _log(b.batch_id,
                 f"CANDIDATES: checkpoint {ckpt_path.name} "
                 f"unreadable ({ckpt_reset['error']}); resume "
                 f"state reset — stage restarts from scratch")

    doc_ids = list(ner_data.keys()) if use_pipeline \
        else list(texts.keys())
    _beat(b, root, "CANDIDATES", len(done_docs), len(doc_ids))
    for idx, doc_id in enumerate(doc_ids):
        if doc_id in done_docs:
            continue
        ents = ner_data.get(doc_id, [])
        text = texts[doc_id]
        sentences = _split_sentences(text)
        # Filter entities by quality BEFORE pairing
        persons = [e["text"] for e in ents
                   if e["label"] == "person" and _is_valid_entity(e["text"])]
        orgs = [e["text"] for e in ents
                if e["label"] == "organization" and _is_valid_entity(e["text"])]
        locs = [e["text"] for e in ents
                if e["label"] == "location" and _is_valid_entity(e["text"])]
        all_ents = persons + orgs + locs
        seen = set()
        doc_cands = []
        doc_blocked = []

        def _match(ent_text, all_ents):
            return next((e for e in all_ents
                         if ent_text.lower() in e.lower() or e.lower() in ent_text.lower()),
                        None)

        def _add(subj, verb, obj, sent, method, extra=None):
            subj_ent = _match(subj, all_ents)
            obj_ent = _match(obj, all_ents)
            if subj_ent and obj_ent and subj_ent.lower() != obj_ent.lower():
                k = (subj_ent.lower(), verb.lower(), obj_ent.lower())
                if k not in seen:
                    seen.add(k)
                    rec = {
                        "doc_id": doc_id, "subject": subj_ent,
                        "relation": verb, "object": obj_ent,
                        "context": sent, "method": method,
                    }
                    if extra:
                        rec.update(extra)
                    doc_cands.append(rec)
                    return True
            return False

        # Phase 1: spaCy on all sentences (fast, local). Queue LLM for failures.
        # Arm B (JEV_RER_V2=1): structure-constrained extractor — candidates
        # carry C7 provenance + C4 asserted flag; C3-blocked triples go to the
        # batch-level blocked list (killed-set audit sidecar).
        use_v2 = os.environ.get("JEV_RER_V2") == "1"
        llm_queue = []  # (sent, s_ents)
        sent_state = {}  # sent index -> rer_found
        for si, sent in enumerate(sentences if use_pipeline else []):
            sent_low = sent.lower()
            rer_found = False
            if nlp and use_v2:
                kept, blocked = extract_triples_constrained(sent, nlp)
                for t in kept:
                    extra = {"path": t["obj_binding"],
                             "subj_source": t["subj_source"],
                             "asserted": t["asserted"]}
                    if _add(t["subject"], t["verb"], t["object"], sent,
                            "rer", extra):
                        rer_found = True
                for bl in blocked:
                    subj_ent = _match(bl["subject"], all_ents)
                    obj_ent = _match(bl["object"], all_ents)
                    if subj_ent and obj_ent and \
                            subj_ent.lower() != obj_ent.lower():
                        doc_blocked.append({
                            "doc_id": doc_id, "subject": subj_ent,
                            "relation": bl["verb"], "object": obj_ent,
                            "context": sent, "method": "rer",
                            "reason": bl["reason"],
                        })
            elif nlp:
                for subj, verb, obj in extract_triples(sent, nlp):
                    if _add(subj, verb, obj, sent, "rer"):
                        rer_found = True
            sent_state[si] = rer_found
            if not rer_found and os.environ.get("JEV_RER_LLM") == "1":
                s_ents = [e for e in all_ents if e.lower() in sent_low][:10]
                if len(s_ents) >= 2:
                    llm_queue.append((si, sent, s_ents))

        # Phase 2: LLM fallback in PARALLEL (16 workers).
        if llm_queue:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            def _llm_job(item):
                si, sent, s_ents = item
                return si, sent, extract_triples_llm_with_usage(
                    sent, s_ents, model=b.config.llm_model)
            with ThreadPoolExecutor(max_workers=16) as ex:
                for fut in as_completed([ex.submit(_llm_job, it) for it in llm_queue]):
                    try:
                        si, sent, (triples, usage) = fut.result()
                    except Exception:
                        # A dead worker used to vanish silently:
                        # its sentence simply got no
                        # LLM pass. Counted in the stage metrics.
                        cand_counters["llm_worker_errors"] += 1
                        continue
                    llm_cost = _conn_add_cost(
                        llm_cost, usage.get("cost", 0.0))
                    for subj, verb, obj in triples:
                        if _add(subj, verb, obj, sent, "rer-llm"):
                            sent_state[si] = True

        # Phase 3: pair fallback for sentences where RER found nothing.
        for si, sent in enumerate(sentences if use_pipeline else []):
            if sent_state.get(si):
                continue
            sent_low = sent.lower()
            s_persons = [p for p in persons if p.lower() in sent_low]
            s_orgs = [o for o in orgs if o.lower() in sent_low]
            s_locs = [l for l in locs if l.lower() in sent_low]
            for p in s_persons[:4]:
                for l in s_locs[:3]:
                    r = "associated_with"
                    k = (p.lower(), r, l.lower(), sent[:80])
                    if k not in seen and p.lower() != l.lower():
                        seen.add(k)
                        doc_cands.append({"doc_id": doc_id, "subject": p,
                                          "relation": r, "object": l,
                                          "context": sent, "method": "pair"})
                for o in s_orgs[:3]:
                    r = "associated_with"
                    k = (p.lower(), r, o.lower(), sent[:80])
                    if k not in seen and p.lower() != o.lower():
                        seen.add(k)
                        doc_cands.append({"doc_id": doc_id, "subject": p,
                                          "relation": r, "object": o,
                                          "context": sent, "method": "pair"})
        # LLM proposer — free triples + clause slots.
        # Proposals pass the deterministic guards in pipeline/proposer.py;
        # survivors become candidates judged by Jev at GATES (no score
        # is assigned here). The pipeline per-doc cap (40) does NOT
        # apply to proposer candidates: their bound is the recipe's
        # per-chunk cap, as in the measured standalone path.
        doc_llm = []
        if use_llm and pair_enum:
            # Pair-enumeration: the roster
            # pair census replaces the free proposer. Same guard,
            # same candidate-record discipline (the guard output
            # IS the candidate set — no parallel second pass);
            # records carry method "pair-enum".
            from . import connections as _conn
            from . import pair_proposer as _pp
            from . import proposer as _prop
            # Named refusal (BindingKeyError) when the proposer
            # seat's key variable is unset — never a bare
            # RuntimeError, and run_batch's preflight normally
            # catches this before any stage runs.
            key = _conn.seat("proposer").require_key()
            res = _pp.propose_pairs_for_doc(
                doc_id, text, roster_by_doc.get(doc_id, []), key,
                model=b.config.llm_model, guidance=guidance,
                relation_inventory=relation_inventory,
                window=pair_window,
                judge_context=pair_judge_context,
                judge_instruction=pair_judge_instruction,
                generic_relation=pair_generic_relation,
                jf1=jf1_on, symmetric_relations=symmetric_set,
                abbrev_guard=pair_abbrev_guard,
                evidence_sentence_in_items=pair_theme_attest)
            proposer_cost = _conn_add_cost(
                proposer_cost, res["cost"])
            proposer_stats["calls"] += res["calls"]
            proposer_stats["errors"] += res["errors"]
            pair_stats["enumerated"] += res["enumerated"]
            pair_stats["answered"] += res["answered"]
            pair_stats["calls"] += res["calls"]
            for ok_, ov in res["outcomes"].items():
                pair_stats["outcomes"][ok_] = pair_stats[
                    "outcomes"].get(ok_, 0) + ov
            # The doc's verdicts join the sidecar pool
            # (they were computed pre-guard, inside the proposer;
            # the guard outcome never alters a verdict record).
            pair_verdicts.extend(res.get("verdicts") or [])
            pair_stats["alias_emitted"] += sum(
                1 for it in res["items"]
                if it["subject"] != it["subject_canonical"]
                or it["object"] != it["object_canonical"])
            kept, blocked = _prop.guard_triples(
                res["items"], text, role_words=role_words,
                entity_types=endpoint_types or None,
                theme_attest_key=("evidence_sentence"
                                  if pair_theme_attest else None),
                quote_punct_insensitive=quote_punct_insensitive)
            for it in kept:
                rec = {"doc_id": doc_id, "subject": it["subject"],
                       "relation": it["relation"],
                       "object": it["object"],
                       "context": _prop.quote_window(text, it["quote"]),
                       "method": "pair-enum", "quote": it["quote"],
                       "attest": it["attest"]}
                for tk in ("subject_type", "object_type"):
                    if it.get(tk):
                        rec[tk] = it[tk]
                doc_llm.append(rec)
            for bl in blocked:
                rec = {"doc_id": doc_id, "subject": bl["subject"],
                       "relation": bl["relation"],
                       "object": bl["object"],
                       "context": _prop.quote_window(text, bl["quote"]),
                       "method": "pair-enum", "quote": bl["quote"],
                       "reason": bl["reason"]}
                for tk in ("subject_type", "object_type"):
                    if bl.get(tk):
                        rec[tk] = bl[tk]
                doc_blocked.append(rec)
        if use_llm and not pair_enum:
            from . import connections as _conn
            from . import proposer as _prop
            # Named refusal (BindingKeyError) — see the pair-enum
            # branch above.
            key = _conn.seat("proposer").require_key()
            res = _prop.propose_triples_for_doc(
                doc_id, text, key, model=b.config.llm_model,
                guidance=guidance, framing=triple_framing)
            proposer_cost = _conn_add_cost(
                proposer_cost, res["cost"])
            proposer_stats["proposed"] += res["proposed"]
            proposer_stats["calls"] += res["calls"]
            proposer_stats["errors"] += res["errors"]
            if endpoint_types:
                proposer_stats["types_dropped_unknown"] += sum(
                    1 for it in res["items"]
                    for tk in ("subject_type", "object_type")
                    if isinstance(it.get(tk), str)
                    and it[tk].strip()
                    and it[tk].strip() not in endpoint_types)
            kept, blocked = _prop.guard_triples(
                res["items"], text, role_words=role_words,
                entity_types=endpoint_types or None,
                quote_punct_insensitive=quote_punct_insensitive,
                theme_attest_chunk=theme_attest_chunk)
            proposer_stats["typed_kept"] += sum(
                1 for it in kept
                if it.get("subject_type") or it.get("object_type"))
            for it in kept:
                rec = {
                    "doc_id": doc_id, "subject": it["subject"],
                    "relation": it["relation"], "object": it["object"],
                    "context": _prop.quote_window(text, it["quote"]),
                    "method": "llm", "quote": it["quote"],
                    "attest": it["attest"]}
                # Validated endpoint types ride the
                # candidate through GATES/ROUTE into WRITE. Absent
                # keys = untyped candidate (legacy record shape).
                for tk in ("subject_type", "object_type"):
                    if it.get(tk):
                        rec[tk] = it[tk]
                doc_llm.append(rec)
            for bl in blocked:
                rec = {
                    "doc_id": doc_id, "subject": bl["subject"],
                    "relation": bl["relation"], "object": bl["object"],
                    "context": _prop.quote_window(text, bl["quote"]),
                    "method": "llm", "quote": bl["quote"],
                    "reason": bl["reason"]}
                for tk in ("subject_type", "object_type"):
                    if bl.get(tk):
                        rec[tk] = bl[tk]
                doc_blocked.append(rec)
            if slot_categories:
                sres = _prop.propose_slots_for_doc(
                    doc_id, text, slot_categories, key,
                    model=b.config.llm_model,
                    framing=clause_framing, rules=clause_rules)
                proposer_cost = _conn_add_cost(
                    proposer_cost, sres["cost"])
                proposer_stats["slots_proposed"] += sres["proposed"]
                proposer_stats["calls"] += sres["calls"]
                proposer_stats["errors"] += sres["errors"]
                fills, sblocked = _prop.guard_slots(sres["fills"], text)
                doc_name = next(
                    (f["value"] for f in fills
                     if f["category"] == "Document Name"), None)
                if not doc_name:
                    stem = Path(source_paths.get(doc_id, "")).name
                    doc_name = stem[:-4] if stem.endswith(".txt") \
                        else (stem or doc_id)
                for f in fills:
                    if f["category"] == "Document Name":
                        continue
                    doc_llm.append({
                        "doc_id": doc_id, "subject": doc_name,
                        "relation": f["category"],
                        "object": f["value"],
                        "context": _prop.quote_window(text, f["quote"]),
                        "method": "clause-llm", "quote": f["quote"]})
                for bl in sblocked:
                    doc_blocked.append({
                        "doc_id": doc_id, "subject": doc_name,
                        "relation": bl["category"],
                        "object": bl["value"],
                        "context": _prop.quote_window(text, bl["quote"]),
                        "method": "clause-llm", "quote": bl["quote"],
                        "reason": bl["reason"]})
        if len(doc_cands) > 40:
            # The per-doc pipeline cap (40) truncates silently by
            # design of the slice; the truncation itself is
            # counted so a capped doc is visible in the
            # stage metrics instead of just producing fewer
            # candidates.
            cand_counters["cap_docs"] += 1
            cand_counters["cap_dropped"] += len(doc_cands) - 40
        capped = doc_cands[:40]
        if use_pipeline and doc_llm:
            # "both": one candidacy per normalized (s, r, o) — the
            # pipeline record wins (it carries the C7 fields Arm C
            # consumes); proposer duplicates are dropped here.
            from .proposer import norm_ws as _nws
            pipe_keys = {(_nws(c["subject"]), _nws(c["relation"]),
                          _nws(c["object"])) for c in capped}
            doc_llm = [c for c in doc_llm
                       if (_nws(c["subject"]), _nws(c["relation"]),
                           _nws(c["object"])) not in pipe_keys]
        candidates.extend(capped)
        candidates.extend(doc_llm)
        blocked_all.extend(doc_blocked)
        # Checkpoint: save progress after each doc
        done_docs.add(doc_id)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(doc_ids):
            with open(d / "candidates.jsonl", "w") as f:
                for c in candidates:
                    f.write(json.dumps(c) + "\n")
            if blocked_all:
                with open(d / "candidates_blocked.jsonl", "w") as f:
                    for bl in blocked_all:
                        f.write(json.dumps(bl) + "\n")
            if pair_verdicts:
                with open(d / "pair_verdicts.jsonl", "w") as f:
                    for v in pair_verdicts:
                        f.write(json.dumps(v) + "\n")
            ckpt_path.write_text(json.dumps({
                "done_docs": sorted(done_docs),
                "proposer_cost_usd": _round_cost(proposer_cost),
                "proposer_stats": proposer_stats,
                "pair_stats": pair_stats}))
            _log(b.batch_id, f"CANDIDATES: {idx+1}/{len(doc_ids)} docs, "
                    f"{len(candidates)} candidates")
            _beat(b, root, "CANDIDATES", len(done_docs), len(doc_ids))
    with open(d / "candidates.jsonl", "w") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    if blocked_all:
        with open(d / "candidates_blocked.jsonl", "w") as f:
            for bl in blocked_all:
                f.write(json.dumps(bl) + "\n")
    if pair_verdicts:
        with open(d / "pair_verdicts.jsonl", "w") as f:
            for v in pair_verdicts:
                f.write(json.dumps(v) + "\n")
    if pair_enum:
        # Sidecar assert: the persisted sidecar
        # must account for every pair judgment the stage metrics
        # record. The sidecar counts verdicts (pre-guard); the
        # metrics count the same outcomes — any drift means the
        # bookkeeping, not the candidates, is broken, and the
        # stage fails loudly rather than recording it silently.
        if pair_verdicts_consistent:
            from collections import Counter
            sidecar_outcomes = dict(Counter(
                v["outcome"] for v in pair_verdicts))
            if sidecar_outcomes != pair_stats["outcomes"]:
                raise RuntimeError(
                    "pair_verdicts sidecar outcome counts "
                    f"{sidecar_outcomes} != pair_enum outcomes "
                    f"{pair_stats['outcomes']}")
        else:
            _log(b.batch_id,
                 "CANDIDATES: pair_verdicts sidecar is partial "
                 "(resumed from a pre-sidecar checkpoint); "
                 "outcome-count assert skipped")
    # Clean up checkpoint
    if ckpt_path.exists():
        ckpt_path.unlink()
    # Metrics by method
    rer_n = sum(1 for c in candidates if c.get("method") in ("rer", "rer-llm"))
    llm_n = sum(1 for c in candidates if c.get("method") == "rer-llm")
    by_method: dict[str, int] = {}
    for c in candidates:
        by_method[c.get("method", "?")] = \
            by_method.get(c.get("method", "?"), 0) + 1
    blocked_reasons: dict[str, int] = {}
    for bl in blocked_all:
        blocked_reasons[bl.get("reason", "?")] = \
            blocked_reasons.get(bl.get("reason", "?"), 0) + 1
    b.metrics["candidates"] = {
        "total": len(candidates), "rer": rer_n,
        "rer_llm": llm_n,
        "pair": len(candidates) - rer_n
        - by_method.get("llm", 0) - by_method.get("clause-llm", 0),
        "rer_llm_cost_usd": _round_cost(llm_cost),
        "source": source, "by_method": by_method,
        "blocked": len(blocked_all), "blocked_reasons": blocked_reasons,
        "proposer": {**proposer_stats,
                     "cost_usd": _round_cost(proposer_cost),
                     "model": b.config.llm_model}}
    # Silent-path counters: see the cand_counters init.
    rer_stats1 = _rer_mod.llm_call_stats()
    b.metrics["candidates"]["rer_llm_calls"] = {
        "attempts": rer_stats1["attempts"] - rer_stats0["attempts"],
        "failures": rer_stats1["failures"] - rer_stats0["failures"]}
    b.metrics["candidates"]["llm_worker_errors"] = \
        cand_counters["llm_worker_errors"]
    b.metrics["candidates"]["candidate_cap"] = {
        "cap": 40, "docs_capped": cand_counters["cap_docs"],
        "candidates_dropped": cand_counters["cap_dropped"]}
    if rer_env["JEV_RER_V2"] or rer_env["JEV_RER_LLM"]:
        # The env toggles change extraction behavior; when either
        # is active the metrics say so (a rerun under different
        # toggles must not be indistinguishable from the default).
        b.metrics["candidates"]["rer_env"] = rer_env
    if ckpt_reset is not None:
        b.metrics["candidates"]["checkpoint_reset"] = ckpt_reset
    if pair_enum:
        b.metrics["candidates"]["pair_enum"] = {
            **pair_stats, "window": pair_window,
            "roster": roster_name,
            "judge_context": pair_judge_context,
            "judge_instruction": pair_judge_instruction,
            "judge_format": pair_judge_format or "legacy"}
        if symmetric_src is not None:
            b.metrics["candidates"]["pair_enum"][
                "symmetric_source"] = symmetric_src
    # Ledger: the stage's actual API spend (usage is returned
    # by extract_triples_llm_with_usage for the fallback and by the
    # proposer for its calls, and summed above). spaCy/pair paths
    # are local and cost 0. Caveat: on a resumed run, FALLBACK spend
    # from the crashed session is NOT re-counted (checkpointed docs
    # are skipped; no durable per-call record exists for it, unlike
    # GATES' queue results). Proposer spend IS checkpointed (above).
    _record_cost(b, "CANDIDATES",
                 _round_cost(_conn_add_cost(llm_cost, proposer_cost)))
    save_batch(b, root)
    advance(b, root, [*stages, "CANDIDATES"])
    _log(b.batch_id, f"CANDIDATES done: {len(candidates)} "
                     f"(by method: {by_method})")
    if blocked_all:
        # Guard blocks are reported BY NAME in the run
        # output, not only in the metrics record — a relation
        # class blocked wholesale (theme_presence) must be
        # visible at the moment it happens, with the sidecar
        # that carries the blocked items.
        _log(b.batch_id,
             f"CANDIDATES blocked: {len(blocked_all)} "
             f"(by reason: {blocked_reasons}) — see "
             f"candidates_blocked.jsonl")


def stage_gates(b: Batch, root: Path, stages: list[str],
                decide_fn) -> None:
    """GATES: Jev noul per candidate (queue-based, resumable).

    Candidates are enqueued as work items. Workers claim, judge, complete.
    A crash mid-stage loses nothing — unclaimed items stay queued.
    """
    from .queue import WorkQueue
    d = batch_dir(root, b.batch_id)
    candidates = [json.loads(l) for l in open(d / "candidates.jsonl")]
    # The $ projection is a planning heuristic, printed only
    # when the judge seat can meter spend at all: on
    # a cost-less binding a dollar figure would fabricate
    # knowledge the product's own doctrine says is unknown —
    # the banner then projects a judgment COUNT instead, and
    # the cap pre-check is skipped (a cap over unmetered
    # spend refuses by name at create/preflight/record).
    from . import connections as _conn_mod
    if _conn_mod.seat("judge").reports_cost:
        projected = len(candidates) * 0.00003  # conservative per-judgment estimate
        if (b.config.max_cost_usd is not None
                and projected > b.config.max_cost_usd):
            raise RuntimeError(
                f"budget cap: projected ${projected:.2f} > max ${b.config.max_cost_usd}")
        projection_note = f"budget ok, projected ${projected:.4f}"
    else:
        projection_note = (f"projected {len(candidates)} judgments "
                           f"(unmetered binding — no $ projection)")

    q = WorkQueue(d / "queue.db")
    worker_id = f"gates-{b.run_id[:8]}"
    try:
        # Enqueue (idempotent — resume won't duplicate)
        items = [{"item_id": f"cand-{i}",
                  "index": i, **c} for i, c in enumerate(candidates)]
        total = q.enqueue(b.batch_id, b.run_id, "GATES", items)
        depth = q.depth(b.batch_id, b.run_id, "GATES")
        pending = depth.get("pending", 0) + depth.get("claimed", 0)
        _log(b.batch_id, f"GATES: {total} queued, {pending} remaining "
                         f"({projection_note}, "
                         f"claim lease {b.config.queue_lease_seconds}s)")
        if pending == 0:
            _log(b.batch_id, "GATES: nothing to do (already complete)")

        def judge(c):
            return decide_fn(c["context"], c["subject"], c["relation"],
                             c["object"], model=b.config.gate_model)

        # Spend already banked by a previous (crashed) session of this
        # stage, so the mid-loop cap check below is resume-correct.
        # The final total is recomputed from all results regardless.
        spent_before = 0.0
        with q._conn:
            cur = q._conn.execute(
                "SELECT result FROM queue_items WHERE batch_id=? "
                "AND run_id=? AND stage='GATES' AND status='done'",
                (b.batch_id, b.run_id))
            for (result_s,) in cur.fetchall():
                if result_s:
                    spent_before = _conn_add_cost(
                        spent_before,
                        json.loads(result_s).get("cost", 0))
        prior_total = (b.metrics.get("cost_ledger") or {}).get(
            "total_usd") or 0.0

        total_cost, errs, done = 0.0, 0, depth.get("done", 0)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as ex:
            while True:
                claimed = q.claim(b.batch_id, b.run_id, "GATES",
                                  worker_id, limit=32,
                                  lease_seconds=b.config.queue_lease_seconds)
                if not claimed:
                    # Crash residue: items still 'claimed' by a dead
                    # worker become claimable only after the queue's
                    # configured lease lapses. Breaking here (the previous
                    # behavior) finalized them as error judgments
                    # (p_true 0.0); wait out the lease instead (§3
                    # resume contract).
                    depth_now = q.depth(b.batch_id, b.run_id, "GATES")
                    if depth_now.get("claimed", 0) \
                            and not depth_now.get("pending", 0):
                        _log(b.batch_id,
                             f"GATES: waiting for "
                             f"{depth_now['claimed']} stale claims to "
                             f"lapse (crash recovery)")
                        _beat(b, root, "GATES", done, total)
                        time.sleep(10)
                        continue
                    break
                futs = {ex.submit(judge, it.payload): it for it in claimed}
                for fut in as_completed(futs):
                    it = futs[fut]
                    try:
                        p_true, usage = fut.result()
                        cost = usage.get("cost")
                        total_cost = _conn_add_cost(total_cost, cost)
                        q.complete(it.item_id,
                                   {"p_true": p_true, "cost": cost},
                                   batch_id=b.batch_id, run_id=b.run_id)
                    except Exception as e:
                        errs += 1
                        st = q.fail(it.item_id, str(e)[:200],
                                    batch_id=b.batch_id, run_id=b.run_id)
                        _log(b.batch_id,
                             f"GATES: item {it.item_id} failed → {st}")
                    done += 1
                    if done % 200 == 0:
                        _log(b.batch_id, f"GATES: {done}/{total}")
                        _beat(b, root, "GATES", done, total)
                # Hard cap (§3): abort the stage rather than silently
                # continue past max_cost_usd. Unmetered spend under
                # a set cap refuses by name — the cap
                # cannot guard what it cannot meter.
                if (b.config.max_cost_usd is not None
                        and (spent_before is None
                             or total_cost is None)):
                    from . import connections as _conn
                    raise _conn.BindingCostError(
                        "GATES recorded unmetered spend (the judge "
                        "binding reports no cost) while max_cost_usd "
                        f"${b.config.max_cost_usd:.2f} is set; a USD "
                        "cap cannot guard spend it cannot meter — "
                        "bind a cost-reporting route, declare "
                        "per-seat rates, or remove the cap "
                        "(--no-cost-cap)")
                if (b.config.max_cost_usd is not None
                        and prior_total + spent_before + total_cost
                        > b.config.max_cost_usd):
                    raise RuntimeError(
                        f"cost cap exceeded in GATES: prior stages "
                        f"${prior_total:.4f} + gates "
                        f"${spent_before + total_cost:.4f} > "
                        f"max_cost_usd ${b.config.max_cost_usd:.2f}")
        # Collect results in index order
        judgments = []
        with q._conn:
            cur = q._conn.execute(
                """SELECT payload, result, status FROM queue_items
                   WHERE batch_id=? AND run_id=? AND stage='GATES'
                   ORDER BY CAST(json_extract(payload, '$.index') AS INTEGER)""",
                (b.batch_id, b.run_id))
            for payload_s, result_s, status in cur.fetchall():
                c = json.loads(payload_s)
                if status == "done" and result_s:
                    r = json.loads(result_s)
                    judgments.append({**c,
                                      "p_true": r["p_true"],
                                      "cost": r["cost"],
                                      "error": None})
                else:
                    judgments.append({**c, "p_true": 0.0, "cost": 0,
                                      "error": f"status={status}"})
        # Recompute cost from results (avoids double-count on resume)
        total_cost = 0.0
        for j in judgments:
            total_cost = _conn_add_cost(total_cost, j.get("cost", 0))
        errs = sum(1 for j in judgments if j.get("error"))
        dlq = q.dlq_items(b.batch_id, "GATES")
        if dlq:
            _log(b.batch_id, f"GATES: {len(dlq)} items in DLQ")
        with open(d / "judgments.jsonl", "w") as f:
            for j in judgments:
                f.write(json.dumps(j) + "\n")
    finally:
        q.close()
    dt = time.perf_counter() - t0
    b.metrics["gates"] = {"judged": len(judgments), "errors": errs,
                          "cost_usd": _round_cost(total_cost),
                          "seconds": round(dt, 1)}
    _record_cost(b, "GATES", _round_cost(total_cost))
    save_batch(b, root)
    advance(b, root, [*stages, "GATES"])
    _log(b.batch_id, f"GATES done: {len(judgments) - errs} ok, "
                     f"cost={_fmt_cost(total_cost)}, {errs} errors")


def _c1_whitelist(setup_cfg: dict | None) -> set | None:
    """C1 (verb-class, preposition) whitelist from the corpus
    profile's setup_config.

    Returns None when the profile declares no whitelist (CUAD:
    c1_whitelist is null with a derivation note) — the caller then
    applies NO C1 demotions and logs that fact; a whitelist is never
    invented at route time (playbook: derive per corpus from the
    first batch's relation census). Accepted shapes: a list of
    [verb, prep] pairs, or c1_profile-style rules
    [{verbs: [...], preps: [...]}]."""
    if not setup_cfg:
        return None
    raw = setup_cfg.get("c1_whitelist")
    if not raw:
        return None
    allow: set = set()
    if isinstance(raw, dict):
        raw = raw.get("rules") or []
    for item in raw:
        if isinstance(item, dict):
            for v in item.get("verbs", []):
                for p in item.get("preps", []):
                    allow.add((v, p))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            allow.add((item[0], item[1]))
    return allow


def stage_route(b: Batch, root: Path, stages: list[str]) -> None:
    """ROUTE: tier decision + review queue, Arm C baseline (§5).

    Design (2026-10-11):
    - Dropped candidates are saved to dropped_queue.jsonl for audit.
    - Human can review false negatives. Nothing is truly lost.

    Arm C demotions (the standalone Arm C route logic,
    wired in): a would-be Tier 1 item carrying C7 fields is demoted
    to Tier 2 — never dropped — when
      C4  asserted is False (negated/modal extraction), or
      C1  its object binding is prep:<lemma> and the (verb, prep)
          pair is not in the corpus profile's C1 whitelist.
    Items without C7 fields (LLM-proposed candidates) carry no
    asserted/path, so neither rule fires on them — their guards ran
    at proposal time (pipeline/proposer.py). C2 theme-presence likewise
    lives at extraction level (§5), not here.
    """
    d = batch_dir(root, b.batch_id)
    judgments = [json.loads(l) for l in open(d / "judgments.jsonl")]
    cfg = b.config
    setup_cfg = _load_setup_config(b, root)
    c1_allow = _c1_whitelist(setup_cfg)
    if c1_allow is None:
        _log(b.batch_id, "ROUTE: no C1 whitelist in profile "
                         "(c1_whitelist null/absent) — Arm C C1 "
                         "demotions skipped, none invented")
    counts = {"tier1": 0, "tier2": 0, "dropped": 0}
    demotes = {"C4": 0, "C1": 0}
    with open(d / "routed.jsonl", "w") as rf, \
            open(d / "review_queue.jsonl", "w") as qf, \
            open(d / "dropped_queue.jsonl", "w") as df:
        for j in judgments:
            p = j.get("p_true", 0.0) or 0.0
            if p >= cfg.review_band_hi:
                tier = 1
            elif p >= cfg.triple_threshold:
                tier = 2
            else:
                tier = 0
            reason = None
            if tier == 1:
                path = j.get("path") or ""
                if j.get("asserted") is False:
                    reason = "C4"
                elif c1_allow is not None and \
                        path.startswith("prep:"):
                    prep = path[5:]
                    if (j.get("relation"), prep) not in c1_allow:
                        reason = "C1"
                if reason:
                    tier = 2
                    demotes[reason] += 1
                    j["armc_demote"] = reason
            if tier == 1:
                j["tier"], j["tag"] = 1, "VERIFIED"
                counts["tier1"] += 1
            elif tier == 2:
                j["tier"], j["tag"] = 2, "UNVERIFIED"
                counts["tier2"] += 1
                qf.write(json.dumps({
                    "batch_id": b.batch_id, "run_id": b.run_id,
                    "doc_id": j["doc_id"], "subject": j["subject"],
                    "relation": j["relation"], "object": j["object"],
                    "p_true": p, "status": "TO_BE_REVIEWED"}) + "\n")
            else:
                j["tier"], j["tag"] = 0, "DROPPED"
                counts["dropped"] += 1
                df.write(json.dumps({
                    "batch_id": b.batch_id, "run_id": b.run_id,
                    "doc_id": j["doc_id"], "subject": j["subject"],
                    "relation": j["relation"], "object": j["object"],
                    "p_true": p, "status": "DROPPED",
                    # Carry the gate error through, so a
                    # dropped row whose gate call FAILED (judgments
                    # row error, e.g. DLQ) is distinguishable from a
                    # judged negative everywhere downstream.
                    "error": j.get("error")}) + "\n")
            rf.write(json.dumps(j) + "\n")
    b.metrics["route"] = {**counts, "armc_demotes": demotes,
                          "route_baseline": "arm_c"}
    _record_cost(b, "ROUTE", 0.0)  # deterministic local routing
    save_batch(b, root)
    advance(b, root, [*stages, "ROUTE"])
    _log(b.batch_id, f"ROUTE done: {counts}, Arm C demotes: {demotes}")


# Cross-document merge guard (option B, decided 2026-10-16): a
# cross-document form-variant pair may auto-merge only when the Jev
# ER gate scores it at or above this p_same. Below it — or unscored —
# the pair is recorded to the review band and never merged.
ER_XDOC_MERGE_MIN = 0.85


def stage_write(b: Batch, root: Path, stages: list[str],
                registry, graph, er_decide_fn=None) -> None:
    """WRITE: registry-first, then graph. (Consistency §10.1)

    Entity resolution (the form-variant guardrail + the
    cross-document merge guard, option B — decided 2026-10-16):
    form variants of the name
    itself merge deterministically WITHIN a document; a cross-
    document form-variant pair may merge only when the Jev ER gate
    scores it p_same >= 0.85 (score recorded on the merge row);
    every other pair goes to the review band — recorded in
    er_decisions with the gate's p_same when consulted, never
    auto-merged (see er_match).
    Relations are normalized at this stage.

    Typed mentions: a candidate may carry
    subject_type / object_type (proposed under the profile's schema
    guidance, validated at CANDIDATES). The mention's type becomes
    its entity's registry label, and the profile's configured
    merge_prohibited_type_pairs bind BOTH ER channels — a
    prohibited pair never merges, within-doc or cross-doc, and is
    recorded to the review band with the prohibition as the
    reason. Untyped mentions / no configured prohibitions =
    the pre-existing behavior exactly.
    """
    from .registry import Entity, Triple
    d = batch_dir(root, b.batch_id)
    tier1 = []
    with open(d / "routed.jsonl") as f:
        for line in f:
            j = json.loads(line)
            if j.get("tier") == 1:
                tier1.append(j)
    _log(b.batch_id, f"WRITE: {len(tier1)} Tier 1 triples")
    # Corpus namespace: the batch's source is its corpus family
    # (§4; same rule RegistryStore.corpus applies to batch.json).
    corpus = b.source
    setup_cfg = _load_setup_config(b, root)
    from .profile_config import prohibited_type_pairs, type_pair_prohibited
    merge_prohibitions = prohibited_type_pairs(setup_cfg)
    if merge_prohibitions:
        _log(b.batch_id,
             f"WRITE: {len(merge_prohibitions)} type-pair merge "
             f"prohibitions active")

    er_cost = 0.0  # Jev ER gate spend, surfaced into the ledger below
    er_review = 0  # pairs recorded to the ER review band (no merge)
    er_gate_merges = 0  # cross-doc merges via the ER gate (>= 0.85)
    er_xdoc_blocked = 0  # cross-doc form-variant pairs held to review
    er_prohibited_blocked = 0  # pairs held by a type prohibition

    def er_match(name: str, label: str, context: str,
                 doc_id: str, mention_type: str = ""
                 ) -> tuple[str, float | None] | None:
        """Resolve a mention to (entity_id, gate_p_same), or None.

        gate_p_same is None for a deterministic within-document
        merge (no gate consulted) and the recorded Jev ER score for
        a cross-document merge the merge guard admitted.

        ER guardrail (§5 — replaces the earlier mechanism whose
        merges measured 0.283 precision) + the cross-document
        merge guard (option B, decided 2026-10-16) + type
        prohibitions:
        0. Type-pair prohibition: when the mention carries a type,
           the candidate entity's stored type (its label) differs,
           and the profile configures that pair in
           merge_prohibited_type_pairs, the pair NEVER merges in
           either channel below — it is recorded to the review band
           with the prohibition as the reason, and scanning
           continues with the next candidate.
        1. Form-variant merge, same document: a candidate whose
           name (or an alias) is a form variant of the mention under
           is_form_variant AND whose home doc (entity.doc_id) is the
           mention's doc merges deterministically — no gate call.
        2. Form-variant, cross-document (or unknown provenance —
           either doc id missing): the Jev ER gate MUST score the
           pair first. p_same >= ER_XDOC_MERGE_MIN auto-merges with
           the score recorded on the merge's er_decisions row;
           below that (or unscored / gate failure) the pair is NEVER
           merged — it is recorded to the review band with its score.
           (The census behind the guard measured unguarded cross-document
           form-variant merges at 0.057 precision.)
        3. Everything else goes to the REVIEW BAND: the pair is
           never auto-merged on similarity. For plausible pairs
           (shared non-generic token; at most 2 per mention) the
           Jev ER gate is consulted and the pair + p_same + model
           are recorded in er_decisions (outcome='review') for
           human adjudication; the mention becomes its own entity.

        Human settlements bind every channel: a pair
        settled keep-separate (`batch er-decide`) is checked
        before the gate in channels (2) and (3) and before the
        within-doc merge in (1) — it is never re-held, never
        re-consulted, and never merged.
        """
        nonlocal er_cost, er_review, er_gate_merges, er_xdoc_blocked
        nonlocal er_prohibited_blocked
        from .registry import _GENERIC_TOKENS, index_normalize, is_form_variant

        def settled_kept_separate(candidate_id: str) -> bool:
            """A pair a human settled keep-separate
            is never re-held — the gate is not re-consulted and
            no new review row is recorded, in either channel
            below. Registries without a decision store settle
            nothing (the interface default)."""
            fn = getattr(registry, "settled_er_decision", None)
            if fn is None:
                return False
            row = fn(name, candidate_id)
            return bool(row) and row.get("outcome") == "kept_separate"
        # Corpus-scoped by default: resolution never crosses namespaces
        # (the cross-corpus name-contamination class). Candidates are
        # looked up UNTYPED (label=""): the registry filters on a
        # non-empty label, which would hide exactly the cross-type
        # candidates the prohibition below must see.
        cands = registry.find_candidates(name, label=label, limit=5,
                                         corpus=corpus)
        if not cands:
            return None
        fv_handled: set[str] = set()  # pairs step (2) already decided
        for c in cands:
            if not (is_form_variant(name, c.name) or any(
                    is_form_variant(name, a) for a in c.aliases)):
                continue
            fv_handled.add(c.entity_id)
            # type-pair prohibition: binds BOTH channels —
            # checked before the within-doc / cross-doc split, the
            # pair never merges and the gate is never consulted.
            # The pair is recorded to the review band with the
            # prohibition as the recorded reason. Either side
            # untyped (empty) -> the prohibition cannot fire.
            if type_pair_prohibited(mention_type, c.label,
                                    merge_prohibitions):
                registry.record_er_decision(
                    b.batch_id, b.run_id, name, c.entity_id, c.name,
                    outcome="review",
                    reason=(f"merge prohibited by profile "
                            f"(merge_prohibited_type_pairs): "
                            f"{mention_type} x {c.label}"))
                er_review += 1
                er_prohibited_blocked += 1
                _log(b.batch_id,
                     f"  ER merge prohibited by profile: '{name}' "
                     f"({mention_type}) vs '{c.name}' ({c.label}) "
                     f"— recorded to review, NOT merged")
                continue
            if settled_kept_separate(c.entity_id):
                _log(b.batch_id,
                     f"  ER pair settled keep-separate (human): "
                     f"'{name}' vs '{c.name}' — not re-held, "
                     f"NOT merged")
                continue
            if doc_id and c.doc_id and doc_id == c.doc_id:
                return c.entity_id, None  # (1) within-doc: as before
            # (2) cross-document / unknown provenance: gate first.
            p_same = None
            if er_decide_fn is not None:
                try:
                    p_same, usage = er_decide_fn(
                        name, c.name, label, context,
                        model=b.config.gate_model)
                    er_cost = _conn_add_cost(
                        er_cost, (usage or {}).get("cost", 0))
                except Exception as e:
                    _log(b.batch_id,
                         f"  ER gate error for cross-doc pair "
                         f"'{name}' vs '{c.name}': {e}")
                    p_same = None
            if p_same is not None and p_same >= ER_XDOC_MERGE_MIN:
                er_gate_merges += 1
                _log(b.batch_id,
                     f"  ER cross-doc gate merge: '{name}' -> "
                     f"'{c.name}' (gate p_same={p_same:.2f} "
                     f">= {ER_XDOC_MERGE_MIN})")
                return c.entity_id, p_same
            registry.record_er_decision(
                b.batch_id, b.run_id, name, c.entity_id, c.name,
                model=b.config.gate_model if p_same is not None else "",
                p_same=p_same, outcome="review")
            er_review += 1
            er_xdoc_blocked += 1
            _log(b.batch_id,
                 f"  ER cross-doc held to review: '{name}' vs "
                 f"'{c.name}' (gate p_same="
                 f"{'unscored' if p_same is None else f'{p_same:.2f}'}, "
                 f"NOT merged)")
        if er_decide_fn is None:
            return None
        name_toks = set(index_normalize(name).split()) - _GENERIC_TOKENS
        consulted = 0
        for c in cands:
            if consulted >= 2:
                break
            if c.entity_id in fv_handled:
                continue  # step (2) scored + recorded this pair already
            cand_toks = set(index_normalize(c.name).split())
            if not (name_toks & cand_toks):
                continue  # index noise, not a plausible pair
            if settled_kept_separate(c.entity_id):
                _log(b.batch_id,
                     f"  ER pair settled keep-separate (human): "
                     f"'{name}' vs '{c.name}' — not re-held, "
                     f"NOT merged")
                continue
            consulted += 1
            try:
                p_same, usage = er_decide_fn(
                    name, c.name, label, context,
                    model=b.config.gate_model)
                er_cost = _conn_add_cost(
                    er_cost, (usage or {}).get("cost", 0))
            except Exception as e:
                _log(b.batch_id, f"  ER review error for '{name}': {e}")
                continue
            registry.record_er_decision(
                b.batch_id, b.run_id, name, c.entity_id, c.name,
                model=b.config.gate_model, p_same=p_same,
                outcome="review")
            er_review += 1
            _log(b.batch_id,
                 f"  ER review band: '{name}' vs '{c.name}' "
                 f"(gate p_same={p_same:.2f}, NOT merged)")
        return None

    # 1. Registry first (source of truth)
    ent_ids: dict[str, str] = {}
    er_merges = 0
    for j in tier1:
        for name, type_key in ((j["subject"], "subject_type"),
                               (j["object"], "object_type")):
            key = name.lower()
            if key in ent_ids:
                continue
            # The mention's validated type (when the
            # candidate carries one) is stored as the entity's
            # registry label and is er_match's prohibition operand.
            # er_match still looks candidates up untyped (label="")
            # exactly as before. Untyped mention -> label "" ->
            # legacy behavior.
            label = j.get(type_key) or ""
            hit = er_match(name, "", j.get("context", "")[:500],
                           j["doc_id"], mention_type=label)
            if hit:
                existing, gate_p = hit
                new_ent = Entity(entity_id="", name=name, label=label,
                                 batch_id=b.batch_id, run_id=b.run_id,
                                 doc_id=j["doc_id"], corpus=corpus)
                if gate_p is None:
                    ent_ids[key] = registry.merge(existing, new_ent)
                else:
                    # Gate-admitted cross-doc merge: the score
                    # is an input of the decision — record it on the
                    # merge's er_decisions row.
                    ent_ids[key] = registry.merge(
                        existing, new_ent, model=b.config.gate_model,
                        p_same=gate_p)
                er_merges += 1
            else:
                ent_ids[key] = registry.create(
                    Entity(entity_id="", name=name, label=label,
                           batch_id=b.batch_id, run_id=b.run_id,
                           doc_id=j["doc_id"], corpus=corpus))
    # 1.4. Store identity: the store holds ONE
    # graph — this build's. When the store's stamp names a
    # different graph, clear it (named, counted) before
    # projecting this one; content no stamp identifies stops
    # the stage by name (the run preflight refuses first —
    # this is the same check at the point of writing). The
    # registry keeps the replaced graph's full history; only
    # the store's current state is replaced.
    store_client = None
    store_replaced = None
    _client = getattr(graph, "c", None)
    if _client is not None:
        from .neo4j_store import Neo4jClient
        if isinstance(_client, Neo4jClient):
            store_client = _client
            from .neo4j_store import (
                batch_graph_identity,
                check_store_write,
                clear_store_graph,
                describe_graph_identity,
            )
            _action, _current = check_store_write(
                store_client, batch_graph_identity(b))
            if _action == "replace" and _current is not None:
                _removed = clear_store_graph(
                    store_client, keep_batch_id=b.batch_id)
                _log(b.batch_id,
                     f"WRITE: replacing the store's current "
                     f"graph ({describe_graph_identity(_current)}) "
                     f"— removed {_removed['rel_edges']} edges, "
                     f"{_removed['entities']} entities, "
                     f"{_removed['chunks']} chunks; the registry "
                     f"keeps the replaced graph's history")
                store_replaced = {
                    "previous_batch": _current["batch_id"],
                    "previous_pack": _current["pack_id"],
                    "previous_pack_version":
                        _current["pack_version"],
                    "previous_triples": _current["triples"],
                    "removed": _removed}
    # 1.5. Project registry entities into the graph. The graph is a
    # projection of the registry (§4): with a file-backed registry the
    # nodes do not exist until projected; with the legacy Neo4j
    # registry create() already wrote them and this idempotent upsert
    # just backfills name/label/aliases/corpus.
    if hasattr(graph, "upsert_entities"):
        records = []
        for eid in sorted(set(ent_ids.values())):
            ent = registry.get(eid)
            if ent is not None:
                records.append({
                    "entity_id": ent.entity_id, "name": ent.name,
                    "label": ent.label, "aliases": ent.aliases,
                    "corpus": ent.corpus or corpus})
        if records:
            graph.upsert_entities(records)
    # 2. Graph second (derived view). Relations are normalized at
    # WRITE: profile vocabulary / explicit map / deterministic
    # canonical form; unmapped surfaces pass through and are counted
    # in relation_map.json for the per-batch report.
    from .relation_map import RelationNormalizer
    normalizer = RelationNormalizer(
        profile_map=(setup_cfg or {}).get("relation_map"),
        vocab=(setup_cfg or {}).get("label_categories") or [])
    triples = []
    for j in tier1:
        stored_rel, _layer = normalizer.normalize(j["relation"])
        triples.append(Triple(
            subject_id=ent_ids[j["subject"].lower()],
            object_id=ent_ids[j["object"].lower()],
            relation=stored_rel, p_true=j["p_true"],
            tier=1, tag="VERIFIED",
            batch_id=b.batch_id, run_id=b.run_id,
            doc_id=j["doc_id"], model=b.config.gate_model))
    n = graph.write_triples(triples)
    # Projection manifest: the exact canonical triples this stage put
    # into the graph. VERIFY reconciles the graph against this file —
    # the WRITE-time ER mapping cannot be reconstructed afterwards
    # from surface names (resolution is order/decision-dependent).
    with open(d / "written_triples.jsonl", "w") as f:
        for t, j in zip(triples, tier1):
            f.write(json.dumps({
                "subject_id": t.subject_id, "relation": t.relation,
                "surface_relation": j["relation"],
                "object_id": t.object_id, "doc_id": t.doc_id}) + "\n")
    rel_report = normalizer.report()
    with open(d / "relation_map.json", "w") as f:
        json.dump(rel_report, f, indent=1)
    # The graph write is complete: stamp the store with this
    # graph's identity — the stamp is what a later
    # build names before replacing this graph, and what
    # serving resolves the store's one graph from.
    if store_client is not None:
        from .neo4j_store import store_inventory, write_graph_stamp
        _inv = store_inventory(store_client)
        write_graph_stamp(store_client, {
            **batch_graph_identity(b),
            "triples": _inv["rel_edges"],
            "entities": _inv["entities"],
            "chunks": _inv["chunks"],
            "stamped_at": datetime.now(UTC).isoformat()})
    b.metrics["write"] = {"triples": n, "entities": len(ent_ids),
                          "er_merges": er_merges,
                          "er_form_variant_merges": er_merges,
                          "er_gate_merges": er_gate_merges,
                          "er_cross_doc_blocked": er_xdoc_blocked,
                          "er_prohibited_blocked": er_prohibited_blocked,
                          "er_review_pairs": er_review,
                          "er_cost_usd": _round_cost(er_cost),
                          "relation_layers": rel_report["layers"],
                          "relations_unmapped":
                              sum(rel_report["unmapped_surfaces"]
                                  .values())}
    if store_replaced is not None:
        b.metrics["write"]["store_replaced"] = store_replaced
    _record_cost(b, "WRITE", _round_cost(er_cost))
    save_batch(b, root)
    advance(b, root, [*stages, "WRITE"])
    _log(b.batch_id, f"WRITE done: {n} triples, {len(ent_ids)} entities, "
                     f"{er_merges} ER merges")
    if er_review:
        # a hold names its remedy on the line that reports
        # it — the settlement flow (er-review prints the decide
        # command in turn), so the reader standing at the WRITE
        # output is never left to hunt for the next step.
        _log(b.batch_id,
             f"WRITE: {er_review} ER pair(s) held for human "
             f"settlement — settle them with: ontosmith batch "
             f"er-review {b.batch_id}")


def _index_embedder(model_name: str):
    """Pick the embedder for INDEX by model name.

    Only hosted provider/model ids (e.g.
    openai/text-embedding-3-large) resolve, through the
    embedder seat's binding. There is no local embedding
    path: the local model measured
    against the large one lost (2026-10-12, measured:
    single-hop recall@10 0.500 -> 0.617 vs MiniLM; see
    rag_eval.py), so a config naming anything else
    refuses here instead of silently embedding with a
    weaker model.
    """
    from .embedder import SeatEmbedder
    if "/" in model_name:
        return SeatEmbedder(model_name)
    raise ValueError(
        f"embed_model {model_name!r} is not a hosted "
        f"provider/model id; the embedder seat binds "
        f"hosted models only (no local embedding) — name "
        f"a provider/model id such as "
        f"openai/text-embedding-3-large")


def stage_index(b: Batch, root: Path, stages: list[str]) -> None:
    """INDEX: Tier 2 chunks → embeddings → vector index (UNVERIFIED)."""
    import pickle
    _log(b.batch_id, "INDEX: building Tier 2 vector index")
    d = batch_dir(root, b.batch_id)
    # Tier 2 = routed tier 2, chunked by doc
    texts = {}
    with open(d / "texts.jsonl") as f:
        for line in f:
            r = json.loads(line)
            texts[r["doc_id"]] = r["text"]
    tier2_docs = set()
    with open(d / "routed.jsonl") as f:
        for line in f:
            j = json.loads(line)
            if j.get("tier") == 2:
                tier2_docs.add(j["doc_id"])
    chunks, meta = [], []
    # Sorted: tier2_docs is a set, and its
    # iteration order leaked into the pickle — the same routed
    # input could produce chunk arrays in different orders across
    # runs. The index content is now a deterministic function of
    # the routed rows.
    for doc_id in sorted(tier2_docs):
        text = texts.get(doc_id, "")
        for i in range(0, len(text), 1000):
            ch = text[i:i + 1000]
            if len(ch.strip()) > 100:
                chunks.append(ch)
                meta.append({"doc_id": doc_id, "chunk_idx": i // 1000,
                             "tag": "UNVERIFIED", "tier": 2,
                             "batch_id": b.batch_id})
    if not chunks:
        _log(b.batch_id, "INDEX: no Tier 2 chunks, skipping")
        b.metrics["index"] = {"chunks": 0}
        _record_cost(b, "INDEX", 0.0)
        save_batch(b, root)
        advance(b, root, [*stages, "INDEX"])
        return
    embedder = _index_embedder(b.config.embed_model)
    embs = embedder.embed(chunks)
    with open(d / "tier2_index.pkl", "wb") as f:
        pickle.dump({"embeddings": embs, "meta": meta,
                     "chunks": chunks,
                     "model": b.config.embed_model}, f)
    emb_usage = embedder.usage
    b.metrics["index"] = {"chunks": len(chunks),
                         "dims": int(embs.shape[1]),
                         "model": b.config.embed_model,
                         "input_tokens": emb_usage["input_tokens"]}
    # Ledger: the embedder's actual spend (SeatEmbedder
    # accumulates usage from each response; token count is recorded
    # in metrics above either way). Cost may be None (unmetered
    # binding) — the ledger records it as unknown, not 0.0.
    _record_cost(b, "INDEX",
                 None if emb_usage["cost"] is None
                 else round(emb_usage["cost"], 4))
    save_batch(b, root)
    advance(b, root, [*stages, "INDEX"])
    _log(b.batch_id, f"INDEX done: {len(chunks)} Tier 2 chunks")


def stage_verify(b: Batch, root: Path, stages: list[str],
                 registry, graph) -> None:
    """VERIFY: consistency checks across registry, graph, and files.

    Checks:
    1. Every graph edge references an existing registry entity.
    2. Tier counts match routed.jsonl.
    3. All Tier 1 edges are tagged VERIFIED.
    4. Review queue count matches Tier 2 count.
    5. Graph projection matches the WRITE manifest exactly
       (batch-wide): the set of (subject_id, relation, object_id)
       edges in the graph equals the set recorded in
       written_triples.jsonl at WRITE time. Catches partial WRITE
       runs and residue from abandoned runs. Batches written before
       the manifest existed skip this check (pass=None).
    6. Store drift spot audit: a deterministic sample of
       manifest attestations at (subject, relation, object, doc_id)
       granularity compared against the store at property level
       (p_true, tier) — sees per-document attestation loss and
       property drift that set equality cannot. pass=None without
       a manifest; with a manifest, a property join of 0
       attestations FAILS the check —
       VERIFIED is never recorded on key existence alone.
    7. Tier-2 index match: tier2_index.pkl is opened and
       its chunk count and embedding model are compared against
       the INDEX metrics; when INDEX recorded 0 chunks the file
       must not exist. pass=None when no INDEX metrics exist.

    Graph checks are scoped by batch_id, NOT run_id: edges legitimately
    carry the id of the run that minted them, so run-scoping made the
    checks vacuous whenever VERIFY ran under a fresh run.
    """
    _log(b.batch_id, "VERIFY: running consistency checks")
    d = batch_dir(root, b.batch_id)
    checks: dict[str, dict] = {}

    # 1. Graph edges → registry entities: every endpoint
    # of every REL edge carrying this batch_id must resolve in the
    # REGISTRY, the source of truth. The previous form compared the
    # graph against its own nodes — an endpoint trivially "exists" as
    # the node its edge is attached to — so the check could not fail
    # even when the registry had never heard of the entity, and the
    # id-less-endpoint count it computed was discarded. Now the
    # endpoints are enumerated from the graph and looked up in the
    # registry one by one; an endpoint with no entity_id, or no
    # registry row, counts as dangling.
    if hasattr(graph, "c"):  # Neo4jGraphStore
        if registry is not None and hasattr(registry, "get"):
            rows = graph.c.run([{"statement":
                "MATCH (a)-[r:REL {batch_id: $b}]->(bb) "
                "RETURN a.entity_id, bb.entity_id",
                "parameters": {"b": b.batch_id}}])[0]
            endpoints = [r["row"] for r in rows]
            seen_ids = {eid for pair in endpoints for eid in pair}
            missing = sorted(eid for eid in seen_ids
                             if eid is not None
                             and registry.get(eid) is None)
            missing_set = set(missing)
            if None in seen_ids:
                missing_set.add(None)
            dangling = sum(1 for s, o in endpoints
                           if s in missing_set or o in missing_set)
            checks["no_dangling_edges"] = {
                "pass": dangling == 0, "dangling": dangling,
                "edges_checked": len(endpoints),
                "missing_endpoints": missing[:10]}
        else:
            checks["no_dangling_edges"] = {
                "pass": None,
                "note": "no registry handle; registry-vs-graph "
                        "comparison not possible"}
    else:
        checks["no_dangling_edges"] = {"pass": None,
                                       "note": "not a Neo4j store"}

    # 2. Tier counts match
    tier_counts = {"tier1": 0, "tier2": 0, "dropped": 0}
    with open(d / "routed.jsonl") as f:
        for line in f:
            t = json.loads(line).get("tier", 0)
            if t == 1:
                tier_counts["tier1"] += 1
            elif t == 2:
                tier_counts["tier2"] += 1
            else:
                tier_counts["dropped"] += 1
    m = b.metrics.get("route", {})
    checks["tier_counts_match"] = {
        "pass": (m.get("tier1") == tier_counts["tier1"]
                 and m.get("tier2") == tier_counts["tier2"]
                 and m.get("dropped") == tier_counts["dropped"]),
        "routed_file": tier_counts, "metrics": m}

    # 3. All Tier 1 edges tagged VERIFIED (batch-wide)
    if hasattr(graph, "c"):
        bad_tags = graph.c.one(
            "MATCH ()-[r:REL {batch_id: $b, tier: 1}]->() "
            "WHERE r.tag <> 'VERIFIED' RETURN count(r)",
            {"b": b.batch_id}) or 0
        checks["tier1_all_verified"] = {"pass": bad_tags == 0,
                                        "bad_tags": bad_tags}

    # 4. Review queue matches Tier 2
    q_count = sum(1 for _ in open(d / "review_queue.jsonl"))
    checks["review_queue_match"] = {
        "pass": q_count == tier_counts["tier2"],
        "queue": q_count, "tier2": tier_counts["tier2"]}

    # 5. Graph projection matches the WRITE manifest (batch-wide).
    manifest_p = d / "written_triples.jsonl"
    if hasattr(graph, "c") and manifest_p.exists():
        expected = set()
        with open(manifest_p) as f:
            for line in f:
                j = json.loads(line)
                expected.add((j["subject_id"], j["relation"],
                              j["object_id"]))
        edge_rows = graph.c.run([{"statement":
            "MATCH (a:Entity)-[r:REL {batch_id: $b}]->(bb:Entity) "
            "RETURN a.entity_id, r.type, bb.entity_id",
            "parameters": {"b": b.batch_id}}])[0]
        actual = {(r["row"][0], r["row"][1], r["row"][2])
                  for r in edge_rows}
        missing = expected - actual
        extra = actual - expected
        checks["graph_projection_match"] = {
            "pass": not missing and not extra,
            "graph_edges": len(actual), "expected_edges": len(expected),
            "missing": len(missing), "extra": len(extra),
            "missing_sample": sorted(missing)[:5],
            "extra_sample": sorted(extra)[:5]}
    elif hasattr(graph, "c"):
        checks["graph_projection_match"] = {
            "pass": None,
            "note": "no written_triples.jsonl (batch predates the "
                    "WRITE manifest); projection not reconciled"}
    else:
        checks["graph_projection_match"] = {
            "pass": None, "note": "not a Neo4j store"}

    # 6. Store drift spot audit. Check 5 compares edge SETS
    #    at (subject, relation, object) granularity; it cannot see
    #    per-document attestation loss (an attestation collapse
    #    that hid 1,478 attestations under near-equal counts) or per-edge property
    #    drift (p_true / tier mutated after WRITE). This check
    #    re-derives expected (subject, type, object, doc_id) rows
    #    from the batch manifest + routed rows and compares a
    #    deterministic sample against the store at property level,
    #    using pipeline/drift_check.py as the single implementation.
    #    pass=None when no manifest exists (same semantics as 5).
    if hasattr(graph, "c") and manifest_p.exists():
        from . import drift_check as dc
        exp, join = dc.expected_from_manifest(
            b.batch_id, root=root, registry=registry)
        act = dc.actual_from_store(b.batch_id, client=graph.c)
        rep = dc.audit(exp, act, sample_n=200)
        # Guard: a property audit that joined
        # 0 attestations compared nothing — the check FAILS and
        # VERIFIED is blocked; it never degrades to a silent pass
        # on key existence alone. (The pre-fix audit resolved the
        # wrong registry + a corpus field save_state never wrote,
        # joined 0 on a real batch, and still recorded VERIFIED.)
        prop_joined = join.get("joined", 0)
        drift_entry = {
            "pass": bool(rep["drift_free"]) and prop_joined > 0,
            "sample_checked": rep["sampled_expected"],
            "missing": rep["missing"],
            "extra": rep["extra"],
            "property_drift": rep["property_drift"],
            "expected_total": rep["expected_total"],
            "actual_total": rep["actual_total"],
            "join": join}
        if prop_joined == 0:
            drift_entry["note"] = (
                "property audit joined 0 attestations — no "
                "p_true/tier comparison was performed, so the "
                "drift check fails rather than passing on key "
                "existence alone")
        checks["drift_spot_audit"] = drift_entry
    elif hasattr(graph, "c"):
        checks["drift_spot_audit"] = {
            "pass": None,
            "note": "no written_triples.jsonl (batch predates "
                    "the WRITE manifest); the property audit did "
                    "not run — no attestation properties were "
                    "compared"}
    else:
        checks["drift_spot_audit"] = {
            "pass": None, "note": "not a Neo4j store"}

    # 7. Tier-2 index artifact vs INDEX metrics: the
    # pickle is opened and its chunk count and embedding model are
    # compared against what INDEX recorded in metrics. Existence of
    # the file alone proved nothing — a stale or foreign index would
    # have passed a stage that never looked inside the artifact.
    idx_m = b.metrics.get("index") or {}
    n_chunks = idx_m.get("chunks")
    pkl_p = d / "tier2_index.pkl"
    if n_chunks is None:
        checks["tier2_index_match"] = {
            "pass": None, "note": "no INDEX metrics recorded"}
    elif n_chunks == 0:
        checks["tier2_index_match"] = {
            "pass": not pkl_p.exists(), "chunks": 0,
            "note": "INDEX recorded 0 chunks; no index file may exist"}
    elif not pkl_p.exists():
        checks["tier2_index_match"] = {
            "pass": False, "chunks": n_chunks,
            "note": "INDEX recorded chunks but tier2_index.pkl "
                    "is missing"}
    else:
        import pickle
        try:
            with open(pkl_p, "rb") as f:
                idx_data = pickle.load(f)
            pkl_chunks = len(idx_data.get("chunks") or [])
            pkl_model = idx_data.get("model")
            checks["tier2_index_match"] = {
                "pass": pkl_chunks == n_chunks
                        and pkl_model == idx_m.get("model"),
                "chunks": n_chunks, "pkl_chunks": pkl_chunks,
                "model": idx_m.get("model"), "pkl_model": pkl_model}
        except Exception as e:
            checks["tier2_index_match"] = {
                "pass": False, "chunks": n_chunks,
                "note": f"tier2_index.pkl unreadable: {e}"}

    all_pass = all(c.get("pass") for c in checks.values()
                   if c.get("pass") is not None)
    b.metrics["verify"] = {"checks": checks, "all_pass": all_pass}
    _record_cost(b, "VERIFY", 0.0)  # local consistency checks
    save_batch(b, root)
    with open(d / "verify_report.json", "w") as f:
        json.dump({"batch_id": b.batch_id, "run_id": b.run_id,
                   "checks": checks, "all_pass": all_pass}, f, indent=1)
    if not all_pass:
        failed = [k for k, v in checks.items() if v.get("pass") is False]
        raise RuntimeError(f"VERIFY failed: {failed}")
    advance(b, root, [*stages, "VERIFY"])
    _log(b.batch_id, f"VERIFY done: all {len(checks)} checks passed")


def _batch_edge_count(graph, batch_id: str) -> int:
    """REL edges in the graph carrying this batch_id (0 when the
    store cannot be queried — purge decisions then stay no-ops)."""
    if graph is None or not hasattr(graph, "c"):
        return 0
    try:
        return graph.c.one(
            "MATCH ()-[r:REL {batch_id: $b}]->() RETURN count(r)",
            {"b": batch_id}) or 0
    except Exception:
        return 0


def purge_batch_attempt(b: Batch, root: Path, graph) -> dict:
    """Retry semantics (decided 2026-10-19): retrying a
    batch that already ran WRITE first DELETES that batch's own
    prior-attempt outputs, then rewrites ("erase, then write again").

    Scoped strictly to this batch_id:
    - graph REL edges carrying this batch_id (all runs — a retry
      mints a fresh run_id, so run-scoped deletion cannot reach
      the prior attempt's edges);
    - this batch's own written_triples.jsonl manifest and
      tier2_index.pkl (a stale manifest/index must never outlive
      the edges it describes).

    Another batch's edges and files are never touched. Entity nodes
    stay: the registry owns them (the rollback
    invariant — rollback deletes edges, not entities). Returns a
    record of what was removed, for batch metrics."""
    d = batch_dir(root, b.batch_id)
    info: dict = {"edges_deleted": 0, "files_removed": []}
    if graph is not None:
        if hasattr(graph, "delete_batch"):
            info["edges_deleted"] = int(
                graph.delete_batch(b.batch_id) or 0)
        elif hasattr(graph, "delete_run"):
            info["edges_deleted"] = int(
                graph.delete_run(b.batch_id, b.run_id) or 0)
        elif hasattr(graph, "_delete_batch_rels"):
            graph._delete_batch_rels(b.batch_id)
            info["edges_deleted"] = -1  # count not reported by store
    for name in ("written_triples.jsonl", "tier2_index.pkl"):
        p = d / name
        if p.exists():
            p.unlink()
            info["files_removed"].append(name)
    return info


def _preflight_bindings(b: Batch, stages: list[str],
                          live_decide: bool) -> None:
    """Refuse before any stage runs when a pending stage needs a
    seat whose key variable is unset.

    The refusal is the connection layer's documented
    :class:`~pipeline.connections.BindingKeyError`, naming the
    seat and the variable — raised here, before INGEST spends
    anything, instead of surfacing mid-run (at CANDIDATES, after
    earlier stages have run, or at the first GATES call) as a
    bare RuntimeError or a late failure. Only seats a *pending*
    stage will actually call are checked: a resumed batch past
    CANDIDATES never re-checks the proposer, and the judge is
    checked only when the caller drives GATES with the live CLI
    decide function (``live_decide``) — stub-driven runs (tests,
    replay) are untouched. The embedder seat is deliberately not
    preflighted: INDEX calls it only when Tier-2 chunks exist,
    and its own call site raises the same named error.
    """
    from . import connections as _conn
    source = (b.config.candidate_source or "pipeline").lower()
    if "CANDIDATES" not in stages and source in ("llm", "both"):
        _conn.seat("proposer").require_key()
    if live_decide and "GATES" not in stages:
        _conn.seat("judge").require_key()


def _preflight_cost_cap(b: Batch, stages: list[str],
                        live_decide: bool) -> None:
    """Stop-line preflight: when a USD cap is set,
    every seat this run will use must be able to meter its
    spend — otherwise the cap could never trip and the run
    refuses by name BEFORE any paid call (the ledger's
    _record_cost applies the same rule at record time).

    Seats by pending stage: CANDIDATES → proposer; GATES →
    judge (live decide only — stub decide functions in
    tests/replays make no provider calls); WRITE → judge via
    the ER decide function (same live gate); INDEX →
    embedder."""
    if b.config.max_cost_usd is None:
        return
    from . import connections as _conn
    seats = []
    if "CANDIDATES" in stages:
        seats.append("proposer")
    if live_decide and ("GATES" in stages or "WRITE" in stages):
        seats.append("judge")
    if "INDEX" in stages:
        seats.append("embedder")
    _conn.require_metered(_conn.resolve_bindings(),
                          b.config.max_cost_usd, seats)


def _preflight_store(b: Batch, stages: list[str], graph) -> None:
    """Store preflight: a build writes into Neo4j —
    a required part of the stack (native graph + vector
    index + full-text in one store), not an optional backend.
    When a graph stage (WRITE / INDEX / VERIFY) is still
    pending and the graph object is the production Neo4j
    store, verify reachability and auth BEFORE anything
    runs — including the retry purge, which also
    touches the store — so an absent or misconfigured store
    refuses by name here instead of killing the batch at
    WRITE after the paid stages have run. Stub graph objects (tests, replay doubles)
    expose no Neo4j client and are untouched.

    Identity: the store holds ONE
    graph, stamped with its identity at WRITE. When this
    build will replace a different graph, the preflight
    names that graph NOW — before the paid stages — and
    when the store holds content no stamp identifies, the
    run stops by name (StoreIdentityError): a build never
    overwrites a graph it cannot name."""
    if graph is None:
        return
    if not [s for s in ("WRITE", "INDEX", "VERIFY")
            if s not in stages]:
        return
    client = getattr(graph, "c", None)
    if client is None:
        return
    from .neo4j_store import StoreUnreachableError, probe
    problem = probe(client)
    if problem is not None:
        raise StoreUnreachableError(
            f"{problem}. A build requires a running Neo4j "
            f"(the graph store) — start one and check "
            f"NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD; "
            f"setup is documented in docs/connections.md "
            f"(Neo4j — the graph store).")
    from .neo4j_store import Neo4jClient
    if isinstance(client, Neo4jClient):
        from .neo4j_store import (
            batch_graph_identity,
            check_store_write,
            describe_graph_identity,
        )
        action, current = check_store_write(
            client, batch_graph_identity(b))
        if action == "replace" and current is not None:
            _log(b.batch_id,
                 f"store: currently holds "
                 f"{describe_graph_identity(current)} — this "
                 f"build replaces it at WRITE (the store "
                 f"holds one graph; the registry keeps the "
                 f"replaced graph's full history)")


def run_batch(b: Batch, root: Path, doc_texts: dict[str, str] | None,
              decide_fn=None, er_decide_fn=None,
              registry=None, graph=None,
              live_decide: bool = False,
              keep_run_id: bool = False) -> Batch:
    """Run all remaining stages. Resumes from checkpoint.

    ``live_decide`` marks that ``decide_fn`` is the CLI's live
    judge function (as opposed to a test/replay stub); it
    enables the binding preflight for the judge seat.

    ``keep_run_id`` is the CLI's --reuse-run: a FAILED batch
    resumes under its existing run_id. This function is the
    single place a resume run_id is minted — callers
    never mint their own, so one resume names exactly one
    run_id."""
    d = batch_dir(root, b.batch_id)
    state_p = d / "state.json"
    stages: list[str] = []
    if state_p.exists():
        stages = json.loads(state_p.read_text()).get("stages_completed", [])
    # Retry rule: a FAILED batch whose failure sits at WRITE or
    # later — WRITE/INDEX in stages_completed, or the stage after
    # the last completed one is WRITE (partial WRITE before the
    # failure was recorded) — is retried by PURGE-THEN-REWRITE:
    # delete this batch's prior-attempt edges and artifacts, roll
    # the checkpoint back to ROUTE, and let WRITE run again from
    # the recorded routed rows. Without the purge, the retry's
    # fresh run_id writes a second edge set beside the first
    # attempt's, and VERIFY (batch-scoped by design) fails on the
    # residue with no self-repair path.
    if b.status == BatchStatus.FAILED and stages and graph is not None \
            and (any(s in stages for s in ("WRITE", "INDEX"))
                 or stages[-1] == "ROUTE"):
        info = purge_batch_attempt(b, root, graph)
        info["purged_run_id"] = b.run_id
        stages = [s for s in stages
                  if s in ("INGEST", "NER", "CANDIDATES", "GATES",
                           "ROUTE")]
        b.metrics["retry_purge"] = info
        save_state(b, root, stages)
        _log(b.batch_id, f"retry purge: {info['edges_deleted']} "
                         f"edges deleted, files {info['files_removed']}; "
                         "checkpoint rolled back to ROUTE")
    # Kill -9 case: the batch sits at ROUTED (checkpoint
    # advanced) but a previous attempt died inside WRITE before any
    # failure could be recorded, leaving partial edges behind.
    # Same rule: purge this batch's edges before WRITE re-enters.
    if b.status == BatchStatus.ROUTED and graph is not None \
            and _batch_edge_count(graph, b.batch_id):
        info = purge_batch_attempt(b, root, graph)
        b.metrics["retry_purge"] = info
        save_batch(b, root)
        _log(b.batch_id, f"retry purge (residue at ROUTED): "
                         f"{info['edges_deleted']} edges deleted, "
                         f"files {info['files_removed']}")
    # Resume from FAILED: map back to the last completed stage's status
    if b.status == BatchStatus.FAILED:
        resume_map = {
            "INGEST": BatchStatus.INGESTED,
            "NER": BatchStatus.NER_DONE,
            "CANDIDATES": BatchStatus.CANDIDATES_DONE,
            "GATES": BatchStatus.GATED,
            "ROUTE": BatchStatus.ROUTED,
            "WRITE": BatchStatus.WRITTEN,
            "INDEX": BatchStatus.INDEXED,
            "VERIFY": BatchStatus.VERIFIED,
        }
        b.status = (resume_map.get(stages[-1], BatchStatus.CREATED)
                    if stages else BatchStatus.CREATED)
        if keep_run_id:
            _log(b.batch_id,
                 f"resuming FAILED batch from {b.status.value} "
                 f"(run_id kept: {b.run_id})")
        else:
            b.run_id = new_run_id()  # failed run is abandoned; new run_id
            _log(b.batch_id,
                 f"resuming FAILED batch from {b.status.value} "
                 f"(new run_id {b.run_id})")
        save_batch(b, root)
    _log(b.batch_id, f"resuming from {b.status.value} "
                     f"(done: {stages})")
    # Provenance: stamp the code version this run executes with. A batch
    # can span several runs (and code fixes between them), so keep the
    # full history, not just the latest hash.
    ver = git_code_version(root)
    if ver:
        if ver != b.config.code_version:
            object.__setattr__(b.config, "code_version", ver)
        if (not b.code_history
                or b.code_history[-1].get("code_version") != ver):
            b.code_history.append({
                "run_id": b.run_id, "code_version": ver,
                "at": datetime.now(UTC).isoformat()})
        save_batch(b, root)
    # Preflight (before the try, so a refusal leaves the batch
    # un-failed and un-started): pending stages' seats must have
    # their keys, or the run refuses with BindingKeyError here.
    _preflight_bindings(b, stages, live_decide and decide_fn is not None)
    # Stop-line preflight, same placement and for
    # the same reason: a USD cap over cost-less seats refuses
    # by name before any stage starts.
    _preflight_cost_cap(b, stages,
                        live_decide and decide_fn is not None)
    # Store preflight, same placement: a build needs
    # Neo4j serving before the first stage runs.
    _preflight_store(b, stages, graph)
    try:
        # Lazy imports for heavy deps
        if b.status == BatchStatus.CREATED:
            assert doc_texts is not None, "doc_texts required for INGEST"
            _check_cost_cap(b)
            _beat(b, root, "INGEST")
            stage_ingest(b, root, doc_texts, stages)
            stages.append("INGEST")
        if b.status == BatchStatus.INGESTED:
            _check_cost_cap(b)
            _beat(b, root, "NER")
            stage_ner(b, root, stages)
            stages.append("NER")
        if b.status == BatchStatus.NER_DONE:
            _check_cost_cap(b)
            _beat(b, root, "CANDIDATES")
            stage_candidates(b, root, stages)
            stages.append("CANDIDATES")
        if b.status == BatchStatus.CANDIDATES_DONE:
            assert decide_fn is not None, "decide_fn required for GATES"
            _check_cost_cap(b)
            _beat(b, root, "GATES")
            stage_gates(b, root, stages, decide_fn)
            stages.append("GATES")
        if b.status == BatchStatus.GATED:
            _check_cost_cap(b)
            _beat(b, root, "ROUTE")
            stage_route(b, root, stages)
            stages.append("ROUTE")
        if b.status == BatchStatus.ROUTED:
            assert registry is not None and graph is not None
            _check_cost_cap(b)
            _beat(b, root, "WRITE")
            stage_write(b, root, stages, registry, graph,
                        er_decide_fn=er_decide_fn)
            stages.append("WRITE")
        if b.status == BatchStatus.WRITTEN:
            _check_cost_cap(b)
            _beat(b, root, "INDEX")
            stage_index(b, root, stages)
            stages.append("INDEX")
        if b.status == BatchStatus.INDEXED:
            _check_cost_cap(b)
            _beat(b, root, "VERIFY")
            stage_verify(b, root, stages, registry, graph)
            stages.append("VERIFY")
        _log(b.batch_id, f"batch run complete at {b.status.value}")
    except Exception as e:
        mark_failed(b, root, stages, str(e)[:500])
        _log(b.batch_id, f"FAILED: {e}")
        raise
    return b
