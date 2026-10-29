"""Relation normalization at WRITE.

Relation strings enter the graph from heterogeneous sources:
free-text verb phrases (LLM proposer), roster-pair relation
names (pair enumeration), and profile category names
(clause-slot proposer). Without a
normalization step the graph is not addressable by a canonical
relation vocabulary and retrieval falls back to string token
overlap (§10b named this a top gap).

Deterministic layers, applied in order at WRITE; the layer that
fired is counted per batch and the per-surface report is written
to the batch dir (relation_map.json):

  vocab       the relation is a member of the corpus profile's
              declared label vocabulary (setup_config
              label_categories) — already canonical, stored verbatim.
  map         the profile/setup config carries an explicit
              relation_map {surface: canonical}; an exact hit on the
              normalized surface stores the mapped canonical form.
  normalized  a built-in deterministic canonicalization fired:
              case/whitespace folding, leading auxiliary/modal
              stripping ("shall pay" -> "pay", "is located in" ->
              "located in"), then space -> "_" joining
              ("located in" -> "located_in").
  surface     nothing fired — the surface form is stored unchanged.
              These are COUNTED and reported per batch (top
              surfaces), never silently dropped: the map grows from
              these reports, per corpus family.
"""
from __future__ import annotations

import re
from collections import Counter

_AUX_PREFIX = {"shall", "will", "must", "may", "might", "is", "are",
               "was", "were", "has", "have", "had", "does", "do",
               "did", "can", "could", "should", "would", "be", "been"}


def _surface_key(relation: str) -> str:
    """Normalized lookup key for map/vocab matching."""
    return re.sub(r"\s+", " ", relation.strip().lower())


def canonical_form(relation: str) -> str:
    """Built-in deterministic canonicalization (layer 'normalized')."""
    toks = _surface_key(relation).strip(" .").split(" ")
    toks = [t for t in toks if t]
    while len(toks) > 1 and toks[0] in _AUX_PREFIX:
        toks = toks[1:]
    return "_".join(toks)


class RelationNormalizer:
    """Stateful per-batch normalizer: counts layers + surfaces."""

    def __init__(self, profile_map: dict | None = None,
                 vocab: set | list | None = None):
        self.profile_map = {_surface_key(k): v
                            for k, v in (profile_map or {}).items()}
        self.vocab = set(vocab or [])
        self.layer_counts: Counter = Counter()
        self.unmapped_surfaces: Counter = Counter()
        self.mapped_pairs: Counter = Counter()

    def normalize(self, relation: str) -> tuple[str, str]:
        """Return (stored_relation, layer)."""
        if relation in self.vocab:
            self.layer_counts["vocab"] += 1
            return relation, "vocab"
        key = _surface_key(relation)
        if key in self.profile_map:
            stored = self.profile_map[key]
            self.layer_counts["map"] += 1
            self.mapped_pairs[f"{relation} -> {stored}"] += 1
            return stored, "map"
        canon = canonical_form(relation)
        if canon and canon != relation:
            self.layer_counts["normalized"] += 1
            self.mapped_pairs[f"{relation} -> {canon}"] += 1
            return canon, "normalized"
        self.layer_counts["surface"] += 1
        self.unmapped_surfaces[relation] += 1
        return relation, "surface"

    def report(self) -> dict:
        return {
            "layers": dict(self.layer_counts),
            "mapped_pairs": dict(self.mapped_pairs.most_common()),
            "unmapped_surfaces": dict(
                self.unmapped_surfaces.most_common()),
        }
