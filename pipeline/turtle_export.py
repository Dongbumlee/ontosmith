"""Turtle export of the Neo4j graph (SPEC requirement).

Pure-Python, no rdflib dependency. Emits:
  - data triples: kg:e_<id> kg:r_<rel> kg:e_<id>  (+ rdfs:label)
  - provenance via RDF reification: rdf:Statement nodes carrying
    kg:p_true, kg:model, kg:gate, kg:doc_id

Usage: NEO4J_PASSWORD=... python -m pipeline.turtle_export --out results/graph.ttl
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

BASE = "http://example.org/ontosmith/"


def sanitize(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "unnamed"


def esc(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/graph.ttl")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--min-p", type=float, default=0.0,
                    help="only export edges with p_true >= MIN_P")
    args = ap.parse_args()

    from neo4j import GraphDatabase
    pw = os.environ.get("NEO4J_PASSWORD")
    if not pw:
        sys.exit("set NEO4J_PASSWORD")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", pw))
    with driver.session() as s:
        rows = list(s.run(
            "MATCH (a:Entity)-[e]->(b:Entity) "
            "WHERE e.p_true >= $minp "
            "RETURN a.name AS s, e.relation AS rel, b.name AS o, "
            "e.p_true AS p, e.model AS model, e.gate AS gate, "
            "e.doc_id AS doc",
            minp=args.min_p))
    driver.close()

    base = args.base
    L = []
    L.append(f"@prefix kg: <{base}> .")
    L.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    L.append("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .")
    L.append("")

    ent_ids: dict[str, str] = {}
    rel_ids: dict[str, str] = {}

    def ent(name: str) -> str:
        if name not in ent_ids:
            eid = "e_" + sanitize(name)
            ent_ids[name] = eid
            L.append(f'kg:{eid} rdfs:label "{esc(name)}" .')
        return ent_ids[name]

    def rel(name: str) -> str:
        if name not in rel_ids:
            rid = "r_" + sanitize(name)
            rel_ids[name] = rid
            L.append(f"kg:{rid} a rdf:Property ;")
            L.append(f'    rdfs:label "{esc(name)}" .')
        return rel_ids[name]

    for i, r in enumerate(rows):
        s_id, p_id, o_id = ent(r["s"]), rel(r["rel"]), ent(r["o"])
        L.append(f"kg:{s_id} kg:{p_id} kg:{o_id} .")
        # reified provenance
        st = f"stmt_{i}"
        L.append(f"kg:{st} a rdf:Statement ;")
        L.append(f"    rdf:subject kg:{s_id} ;")
        L.append(f"    rdf:predicate kg:{p_id} ;")
        L.append(f"    rdf:object kg:{o_id} ;")
        prov = [f'    kg:p_true "{r["p"]}"']
        if r["model"]:
            prov.append(f'    kg:model "{esc(str(r["model"]))}"')
        if r["gate"]:
            prov.append(f'    kg:gate "{esc(str(r["gate"]))}"')
        if r["doc"]:
            prov.append(f'    kg:doc_id "{esc(str(r["doc"]))}"')
        L.append(" ;\n".join(prov) + " .")

    out = Path(args.out)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Wrote {out}: {len(rows)} edges, {len(ent_ids)} entities, "
          f"{len(rel_ids)} relations")


if __name__ == "__main__":
    main()
