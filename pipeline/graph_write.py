"""Write gate-accepted triples into Neo4j with full provenance.

Usage:
  NEO4J_PASSWORD=... python -m pipeline.graph_write \
      --results results/scale1k_jev-1.13_noul_v2.json --threshold 0.25

Schema:
  (:Entity {name}) -[r:RELATION {p_true, model, gate, doc_id, label}]-> (:Entity {name})
Relation labels are sanitized to UPPER_SNAKE; the original label is kept in
r.relation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def sanitize(rel: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", rel).strip("_").upper()
    return s or "RELATED_TO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    ap.add_argument("--wipe", action="store_true", help="delete all before writing")
    ap.add_argument("--model", default=None)
    ap.add_argument("--gate", default=None)
    args = ap.parse_args()
    if not args.password:
        sys.exit("set NEO4J_PASSWORD")

    d = json.loads(Path(args.results).read_text())
    stem = Path(args.results).stem  # e.g. scale1k_jev-1.13_noul_v2
    rest = stem.split("scale1k_")[-1]
    if rest.startswith("jev-1.13"):
        model, gate = "typesafe/jev-1.13", rest[len("jev-1.13") + 1:]
    elif rest.startswith("microsoft-decision-1"):
        model, gate = "microsoft/microsoft-decision-1", rest[len("microsoft-decision-1") + 1:]
    else:
        model, gate = "unknown", "unknown"
    model = args.model or model
    gate = args.gate or gate
    accepted = [r for r in d["records"] if r["p_true"] >= args.threshold]
    print(f"{len(accepted)}/{len(d['records'])} accepted at t={args.threshold}")

    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("graph_write needs the neo4j driver (the [neo4j] "
              "extra): pip install ontosmith[neo4j]",
              file=sys.stderr)
        sys.exit(2)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    with driver.session() as s:
        if args.wipe:
            s.run("MATCH (n) DETACH DELETE n")
        n_edges = 0
        for r in accepted:
            rel = sanitize(r["relation"])
            s.run(
                "MERGE (a:Entity {name: $sa}) "
                "MERGE (b:Entity {name: $ob}) "
                f"MERGE (a)-[e:{rel}]->(b) "
                "SET e.p_true=$p, e.model=$model, e.gate=$gate, "
                "    e.doc_id=$doc, e.label=$label, e.relation=$relation",
                sa=r["subject"], ob=r["object"], p=r["p_true"],
                model=model, gate=gate, doc=r["doc_id"],
                label=r["label"], relation=r["relation"],
            )
            n_edges += 1
        nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        edges = s.run("MATCH ()-[e]->() RETURN count(e) AS c").single()["c"]
    driver.close()
    print(f"graph now: {nodes} entities, {edges} relations "
          f"({n_edges} edge-writes this run)")
    # graph precision: fraction of written edges that are gold (label==1)
    prec = sum(1 for r in accepted if r["label"] == 1) / len(accepted)
    print(f"graph precision (gold labels): {prec:.3f}")


if __name__ == "__main__":
    main()
