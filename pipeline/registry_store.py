"""RegistryStore — file-backed reads over registry / batch artifacts.

The registry is the source of truth and
Neo4j is a rebuildable projection of it. In the original layout the registry's
durable form is the batch artifact tree (`batches/<batch_id>/`:
batch.json, routed.jsonl, written_triples.jsonl, ner.jsonl,
texts.jsonl, tier2_index.pkl). This module is the ONLY file-side
reader the storage layer uses, so the deployable RegistryStore
(SQLite/PostgreSQL behind the same reads, §4) can replace it without
touching GraphStore.

Read-only: nothing here writes to the artifact tree.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path


class RegistryStore:
    """File-backed registry reads (file-backed form of the RegistryStore)."""

    def __init__(self, root: Path | None = None):
        # Default root: the batch state root (batch_cli.state_root
        # — $ONTOSMITH_ROOT, else the source tree, else the working
        # directory), imported lazily. The state root is a
        # workspace location; an installed copy's package tree
        # holds no batches/ tree and is never the default.
        if root is None:
            from .batch_cli import state_root
            root = state_root()
        self.root = Path(root)
        self.batches = self.root / "batches"

    def batch_dir(self, batch_id: str) -> Path:
        return self.batches / batch_id

    # -- batch manifest -------------------------------------------------
    def batch_meta(self, batch_id: str) -> dict:
        """batch.json: batch_id, run_id, source, collection, doc_ids,
        config, metrics."""
        p = self.batch_dir(batch_id) / "batch.json"
        return json.loads(p.read_text())

    def corpus(self, batch_id: str) -> str:
        """Corpus namespace for the batch (§4): the batch.json source."""
        return self.batch_meta(batch_id).get("source") or batch_id

    def run_id(self, batch_id: str) -> str:
        return self.batch_meta(batch_id).get("run_id", "")

    def doc_ids(self, batch_id: str) -> list[str]:
        return list(self.batch_meta(batch_id).get("doc_ids", []))

    # -- documents --------------------------------------------------------
    def documents(self, batch_id: str) -> list[dict]:
        """[{doc_id, source_path}] from texts.jsonl (streamed; the text
        itself is not loaded). Empty list when texts.jsonl is absent —
        callers fall back to the tier2 index metadata."""
        p = self.batch_dir(batch_id) / "texts.jsonl"
        if not p.exists():
            return []
        out = []
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                out.append({"doc_id": r["doc_id"],
                            "source_path": r.get("source_path", "")})
        return out

    # -- triples ----------------------------------------------------------
    def manifest_triples(self, batch_id: str) -> list[dict] | None:
        """written_triples.jsonl — the WRITE manifest: the exact
        canonical triples (subject_id, relation, object_id, doc_id)
        the batch's WRITE stage put into the graph. None for batches
        that predate the manifest (their projection cannot be
        reconciled from artifacts; see batch_run.stage_verify)."""
        p = self.batch_dir(batch_id) / "written_triples.jsonl"
        if not p.exists():
            return None
        with open(p) as f:
            return [json.loads(line) for line in f]

    def routed_rows(self, batch_id: str, tier: int | None = None) -> list[dict]:
        """routed.jsonl rows (surface names + gate provenance),
        optionally filtered to one tier."""
        p = self.batch_dir(batch_id) / "routed.jsonl"
        out = []
        with open(p) as f:
            for line in f:
                j = json.loads(line)
                if tier is None or j.get("tier") == tier:
                    out.append(j)
        return out

    # -- NER mentions -------------------------------------------------------
    def ner_mentions(self, batch_id: str) -> dict[str, list[dict]]:
        """ner.jsonl → {doc_id: [{text, label, score}]} (surface
        mentions per document, no offsets — the NER stage records
        document-level mentions)."""
        p = self.batch_dir(batch_id) / "ner.jsonl"
        out: dict[str, list[dict]] = {}
        if not p.exists():
            return out
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                out[r["doc_id"]] = r.get("entities", [])
        return out

    # -- serving index (E5 input) -------------------------------------------
    def tier2_index(self, batch_id: str) -> dict | None:
        """tier2_index.pkl → {embeddings (n, dims) ndarray, meta
        [{doc_id, chunk_idx, tag, tier, batch_id}], chunks [text],
        model}. None when the batch has no INDEX output."""
        p = self.batch_dir(batch_id) / "tier2_index.pkl"
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)
