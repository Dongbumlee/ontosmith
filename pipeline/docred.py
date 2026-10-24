"""DocRED loading, sampling, and candidate construction for v0.

v0 candidates:
  positives = gold triples from the DocRED dev set
  negatives = corrupted variants (same entity pair, wrong relation; or a
              relation swapped between two gold triples of the same doc)

DocRED relation ids are mapped to human-readable labels via rel_info.json.
"""
from __future__ import annotations

import gzip
import json
import random
import shutil
import urllib.request
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/thunlp/docred/resolve/main/data"
DEV_URL = HF_BASE + "/dev.json.gz"
REL_INFO_URL = HF_BASE + "/rel_info.json.gz"


def data_dir() -> Path:
    """Where the downloaded DocRED corpus lives: the state
    root's data/ directory (batch_cli.state_root, imported
    lazily) — the operator workspace, never the installed
    package tree."""
    from .batch_cli import state_root
    return state_root() / "data"


def _download(url: str, dest: Path) -> None:
    """Download a .gz URL and store the decompressed JSON at dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    tmp_gz = dest.with_suffix(dest.suffix + ".gz.tmp")
    urllib.request.urlretrieve(url, tmp_gz)
    with gzip.open(tmp_gz, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    tmp_gz.unlink()


def load_dev() -> tuple[list[dict], dict[str, str]]:
    d = data_dir()
    _download(DEV_URL, d / "dev.json")
    _download(REL_INFO_URL, d / "rel_info.json")
    docs = json.loads((d / "dev.json").read_text())
    rel_info = json.loads((d / "rel_info.json").read_text())
    return docs, rel_info


def doc_text(doc: dict) -> str:
    return " ".join(" ".join(sent) for sent in doc["sents"])


def entity_name(doc: dict, eid: int) -> str:
    return doc["vertexSet"][eid][0]["name"]


def build_candidates(
    docs: list[dict],
    rel_info: dict[str, str],
    n_docs: int,
    seed: int = 42,
) -> list[dict]:
    """Return a list of candidate dicts with gold labels.

    Each candidate: {doc_id, text, subject, relation, object, label}
    label is 1 for gold triples, 0 for corrupted negatives.
    """
    rng = random.Random(seed)
    sampled = rng.sample(docs, min(n_docs, len(docs)))
    rel_ids = list(rel_info.keys())
    candidates: list[dict] = []

    for doc in sampled:
        text = doc_text(doc)
        gold_triples: list[tuple[str, str, str]] = []
        for label in doc.get("labels", []):
            h = entity_name(doc, label["h"])
            t = entity_name(doc, label["t"])
            r = rel_info.get(label["r"], label["r"])
            gold_triples.append((h, r, t))
            candidates.append(
                {
                    "doc_id": doc["title"],
                    "text": text,
                    "subject": h,
                    "relation": r,
                    "object": t,
                    "label": 1,
                }
            )
        # negatives: for each gold triple, corrupt the relation
        for h, r, t in gold_triples:
            wrong = rng.choice(rel_ids)
            wrong_label = rel_info.get(wrong, wrong)
            if wrong_label == r:
                continue
            candidates.append(
                {
                    "doc_id": doc["title"],
                    "text": text,
                    "subject": h,
                    "relation": wrong_label,
                    "object": t,
                    "label": 0,
                }
            )
    rng.shuffle(candidates)
    return candidates
