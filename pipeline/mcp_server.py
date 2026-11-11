"""MCP server: ``ontosmith mcp`` — the built graph,
served to AI callers over the Model Context Protocol
(design: mcp-interface-design.md; REQUIRED for 0.2.0).

Five read-only tools — ``search``, ``entity``,
``neighborhood``, ``connect``, ``triple`` — over the
same serving store ``ontosmith serve`` opens (``--source``
/ ``--batch`` / ``--store``; the shipped example build
when neither is given). Building stays a human-run CLI
process; this surface only serves a finished build.

Every tool returns the one envelope
(:mod:`pipeline.mcp_envelope`): hoisted ``docs``, items
with citation keys, a hard token budget with signalled
truncation, and honest totals. Evidence is not optional:
a triple and its evidence sentences travel in one
object, conflicts are flagged rather than resolved,
and derived facts are marked as derived. There is no
``ask`` tool — the caller's model composes answers from
returned evidence — and no query language crosses the
boundary; parameters are structured only.

Transport is stdio (local agents first, per design).
The MCP SDK is an optional extra (``ontosmith[mcp]``);
the service layer below is SDK-free and testable
without it — only :func:`build_server` and :func:`main`
touch the SDK.
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

from pipeline import mcp_envelope as env
from pipeline import search as _search

#: Evidence sentences attached to a triple in list
#: contexts (search / neighborhood / connect steps).
#: The ``triple`` tool returns the full set.
EVIDENCE_CAP = 3


class McpToolError(ValueError):
    """A corrective refusal: the message names the
    valid range or the right tool, so the calling model
    can fix its call in the same session (design §6)."""


def _check_int(name: str, val, lo: int, hi: int) -> int:
    """Range-check one integer parameter; the refusal
    names the parameter, its purpose range, and the
    offending value."""
    try:
        val = int(val)
    except (TypeError, ValueError):
        raise McpToolError(
            f"parameter {name} must be an integer, "
            f"got {val!r}") from None
    if not lo <= val <= hi:
        raise McpToolError(
            f"parameter {name} must be {lo}..{hi}, "
            f"got {val}")
    return val


class McpService:
    """The five tools' logic over one opened serving
    store. Transport-free: the FastMCP shell below is a
    thin adapter, so the contract is testable without
    the SDK or a socket."""

    def __init__(self, store, embedder=None):
        self.store = store
        self.embedder = embedder
        self._records: list[dict] | None = None
        self._by_key: dict[tuple, list[dict]] | None = None
        self._objects_by_sr: dict[tuple, set] | None = None

    # -- record indexes (built once, lazily) -----------

    def _index(self) -> None:
        if self._by_key is not None:
            return
        self._records = self.store.triples()
        by_key: dict[tuple, list[dict]] = {}
        objects_by_sr: dict[tuple, set] = {}
        for rec in self._records:
            key = (rec["subject_id"], rec["relation"],
                   rec["object_id"])
            by_key.setdefault(key, []).append(rec)
            objects_by_sr.setdefault(
                (rec["subject_id"], rec["relation"]),
                set()).add(rec["object_id"])
        self._by_key = by_key
        self._objects_by_sr = objects_by_sr

    def _conflicts(self, key: tuple) -> list[str]:
        """Triple ids that conflict with ``key``: same
        subject + relation, a different object. Both
        sides are returned flagged; picking a winner is
        the caller's judgment (design §7)."""
        self._index()
        sid, rel, oid = key
        others = self._objects_by_sr.get((sid, rel), set())
        return sorted(
            env.triple_id(sid, rel, other)
            for other in others if other != oid)

    def _triple_item(self, key: tuple,
                     builder: env.EnvelopeBuilder,
                     *, evidence_cap: int = EVIDENCE_CAP,
                     score: float | None = None,
                     arm: str | None = None) -> dict:
        """One claim–evidence object (design §7): the
        triple, its evidence sentences (capped in list
        contexts), its citation keys, and its conflict
        flags — never a bare triple."""
        self._index()
        records = self._by_key[key]
        head = records[0]
        evidence = []
        for rec in records[:evidence_cap]:
            ref = builder.doc_ref(rec["doc_id"])
            evidence.append({
                "text": rec["quote"],
                "doc": ref,
                "cite": builder.cite_key(ref),
            })
        item: dict = {
            "kind": "triple",
            "triple_id": env.triple_id(*key),
            "subject": head["subject"],
            "relation": head["relation"],
            "object": head["object"],
            "subject_id": key[0],
            "object_id": key[2],
            "evidence": evidence,
        }
        if len(records) > evidence_cap:
            item["evidence_total"] = len(records)
        if score is not None:
            item["score"] = score
        if arm is not None:
            item["arm"] = arm
        conflicts = self._conflicts(key)
        if conflicts:
            item["conflicts_with"] = conflicts
        return item

    # -- tools ------------------------------------------

    def search(self, query: str, k: int = 8,
               max_tokens: int = 2000,
               cursor: str | None = None) -> dict:
        """Hybrid retrieval: graph-arm triples (claims
        with evidence) interleaved by rank with the
        document arm's text evidence."""
        if not (query or "").strip():
            raise McpToolError(
                "parameter query must be a non-empty "
                "string: state the fact or relation you "
                "are looking for")
        k = _check_int("k", k, 1, 50)
        max_tokens = _check_int(
            "max_tokens", max_tokens, 200, 20000)
        offset = env.decode_cursor(cursor)
        try:
            res = _search.hybrid_search(
                self.store, query, k=k,
                embedder=self.embedder)
        except ValueError as e:
            raise McpToolError(str(e)) from e
        builder = env.EnvelopeBuilder()
        triple_items: list[dict] = []
        for rec in res["triples"]:
            key = (rec["subject_id"], rec["relation"],
                   rec["object_id"])
            triple_items.append(self._triple_item(
                key, builder, score=rec.get("score"),
                arm="graph"))
        text_items: list[dict] = []
        text_arm = ("vector" if "vector" in res["arms"]
                    else "lexical")
        for doc in res["documents"]:
            for chunk in doc["chunks"]:
                ref = builder.doc_ref(doc["doc_id"])
                text_items.append({
                    "kind": "text",
                    "text": chunk["text"],
                    "doc": ref,
                    "cite": builder.cite_key(ref),
                    "score": doc["score"],
                    "arm": text_arm,
                })
        # Cross-arm dedup (design §7): a chunk whose
        # text carries a returned triple's evidence
        # sentence is the same evidence — the triple
        # item keeps it, marked "both", and the chunk
        # item is dropped.
        quotes = {
            ev["text"] for item in triple_items
            for ev in item["evidence"]}
        kept_text = []
        for item in text_items:
            carriers = [q for q in quotes
                        if q and q in item["text"]]
            if carriers:
                for t_item in triple_items:
                    if any(ev["text"] in carriers
                           for ev in t_item["evidence"]):
                        t_item["arm"] = "both"
                continue
            kept_text.append(item)
        # Answer-shaped order: interleave the two arms
        # by rank, each arm's internal order preserved.
        merged: list[dict] = []
        for i in range(max(len(triple_items),
                           len(kept_text))):
            if i < len(triple_items):
                merged.append(triple_items[i])
            if i < len(kept_text):
                merged.append(kept_text[i])
        total = len(merged)
        out = builder.build(
            merged[offset:], total=total, offset=offset,
            max_tokens=max_tokens)
        out["arms"] = res["arms"]
        if res["entities"]:
            out["linked_entities"] = [
                {"entity_id": e["entity_id"],
                 "name": e["name"]}
                for e in res["entities"]]
        if res["notes"]:
            out["notes"] = res["notes"]
        return out

    def entity(self, name: str) -> dict:
        """Resolve a name to the canonical entity.
        Ambiguity returns candidates with distinguishers
        — the server never silently picks one (design
        §2). Free-text names are accepted here only;
        every other tool takes ids (design §6)."""
        if not (name or "").strip():
            raise McpToolError(
                "parameter name must be a non-empty "
                "string")
        hits = self.store.find_entities(name, limit=5)
        builder = env.EnvelopeBuilder()
        if not hits:
            out = builder.build([], total=0)
            out["resolved"] = False
            return out
        exact = [h for h in hits
                 if h["name"].lower() == name.lower()]
        if len(hits) > 1 and not exact:
            items = []
            for h in hits:
                as_subj, as_obj = self._counts(
                    h["entity_id"])
                items.append({
                    "kind": "entity_candidate",
                    "entity_id": h["entity_id"],
                    "name": h["name"],
                    "distinguisher":
                        f"{as_subj + as_obj} triples "
                        f"endpoint this entity",
                })
            out = builder.build(items, total=len(items))
            out["resolved"] = False
            out["ambiguous"] = True
            return out
        ent = exact[0] if exact else hits[0]
        as_subj, as_obj = self._counts(ent["entity_id"])
        item = {
            "kind": "entity",
            "entity_id": ent["entity_id"],
            "name": ent["name"],
            "triples_as_subject": as_subj,
            "triples_as_object": as_obj,
        }
        if ent.get("aliases"):
            item["aliases"] = list(ent["aliases"])
        if ent.get("type"):
            item["type"] = ent["type"]
        out = builder.build([item], total=1)
        out["resolved"] = True
        return out

    def _counts(self, entity_id: str) -> tuple[int, int]:
        self._index()
        as_subj = sum(
            len(recs) for (sid, _, _), recs
            in self._by_key.items() if sid == entity_id)
        as_obj = sum(
            len(recs) for (_, _, oid), recs
            in self._by_key.items() if oid == entity_id)
        return as_subj, as_obj

    def _entity_or_refuse(self, entity_id: str) -> dict:
        ent = self.store.entity(entity_id)
        if ent is None:
            raise McpToolError(
                f"no entity with id {entity_id!r}: "
                "resolve the name with the entity tool "
                "first, then pass its entity_id")
        return ent

    def neighborhood(self, entity_id: str,
                     relation: str | None = None,
                     depth: int = 1, limit: int = 20,
                     max_tokens: int = 2000,
                     cursor: str | None = None) -> dict:
        """One entity's relations as claim–evidence
        triple items (depth 1; 2 widens to the
        neighbours' relations)."""
        self._entity_or_refuse(entity_id)
        depth = _check_int("depth", depth, 1, 2)
        limit = _check_int("limit", limit, 1, 100)
        max_tokens = _check_int(
            "max_tokens", max_tokens, 200, 20000)
        offset = env.decode_cursor(cursor)
        nb = self.store.neighborhood(
            [entity_id], hops=depth)
        keys: list[tuple] = []
        seen: set[tuple] = set()
        for edge in nb["edges"]:
            key = (edge["from"], edge["type"], edge["to"])
            if key in seen:
                continue
            seen.add(key)
            if relation and edge["type"] != relation:
                continue
            keys.append(key)
        builder = env.EnvelopeBuilder()
        items = [self._triple_item(key, builder)
                 for key in keys]
        total = len(items)
        window = items[offset:offset + limit]
        out = builder.build(
            window, total=total, offset=offset,
            max_tokens=max_tokens)
        if offset + len(window) < total and \
                not out["truncated"]:
            out["truncated"] = True
            out["cursor"] = env.encode_cursor(
                offset + len(window))
        out["entity_id"] = entity_id
        return out

    def connect(self, from_id: str, to_id: str,
                max_hops: int = 3, limit: int = 5,
                max_tokens: int = 2000) -> dict:
        """Shortest evidence-backed paths between two
        entities. A path is a chain of evidenced triples
        — not a claim that the endpoints are directly
        related (design §7). No path is a state, not an
        error."""
        self._entity_or_refuse(from_id)
        self._entity_or_refuse(to_id)
        max_hops = _check_int("max_hops", max_hops, 1, 4)
        limit = _check_int("limit", limit, 1, 20)
        max_tokens = _check_int(
            "max_tokens", max_tokens, 200, 20000)
        self._index()
        adjacency: dict[str, list[tuple]] = {}
        for key in self._by_key:
            sid, _, oid = key
            adjacency.setdefault(sid, []).append(
                (oid, key))
            adjacency.setdefault(oid, []).append(
                (sid, key))
        # BFS for the shortest length, then enumerate
        # shortest paths only, capped at ``limit``.
        dist = {from_id: 0}
        queue = deque([from_id])
        while queue:
            node = queue.popleft()
            if dist[node] >= max_hops:
                continue
            for nxt, _ in adjacency.get(node, []):
                if nxt not in dist:
                    dist[nxt] = dist[node] + 1
                    queue.append(nxt)
        builder = env.EnvelopeBuilder()
        if to_id not in dist:
            return builder.build([], total=0)
        shortest = dist[to_id]
        paths: list[list[tuple]] = []

        def walk(node: str, used: set,
                 acc: list[tuple]) -> None:
            if len(paths) >= limit:
                return
            if node == to_id and len(acc) == shortest:
                paths.append(list(acc))
                return
            if len(acc) >= shortest:
                return
            for nxt, key in adjacency.get(node, []):
                if nxt in used:
                    continue
                # Stay on a shortest-path DAG: every
                # step grows the BFS distance by one
                # (to_id sits at distance == shortest,
                # so the final hop satisfies this too).
                if dist.get(nxt) != len(acc) + 1:
                    continue
                used.add(nxt)
                acc.append(key)
                walk(nxt, used, acc)
                acc.pop()
                used.discard(nxt)

        walk(from_id, {from_id}, [])
        items = []
        for path in paths:
            steps = [self._triple_item(
                key, builder, evidence_cap=1)
                for key in path]
            items.append({
                "kind": "path",
                "hops": len(path),
                "steps": steps,
            })
        return builder.build(
            items, total=len(items),
            max_tokens=max_tokens)

    def triple(self, triple_id: str) -> dict:
        """Audit one triple: the full evidence set and
        the provenance chain behind it."""
        self._index()
        target = None
        for key in self._by_key:
            if env.triple_id(*key) == triple_id:
                target = key
                break
        if target is None:
            raise McpToolError(
                f"no triple with id {triple_id!r}: pass "
                "a triple_id returned by search, "
                "neighborhood, or connect")
        builder = env.EnvelopeBuilder()
        records = self._by_key[target]
        head = records[0]
        item = self._triple_item(
            target, builder,
            evidence_cap=len(records))
        provenance = {}
        for field in ("batch_id", "tier", "tag",
                      "model", "gate"):
            vals = {rec.get(field) for rec in records}
            vals.discard(None)
            vals.discard("")
            if vals:
                provenance[field] = (
                    sorted(vals, key=str)[0]
                    if len(vals) == 1
                    else sorted(vals, key=str))
        if provenance:
            item["provenance"] = provenance
        item["p_true"] = head.get("p_true")
        return builder.build([item], total=1)


# ----------------------------------------------------------------------
# FastMCP shell — the only SDK-touching layer
# ----------------------------------------------------------------------

_SEARCH_DESC = (
    "Search the built knowledge graph for evidence "
    "about a topic, fact, or relation. Returns ranked "
    "evidence: graph triples (each a claim bound to "
    "its evidence sentences) interleaved with matching "
    "document passages. Use this first for fact "
    "questions. Do NOT use it to enumerate everything "
    "about one known entity — resolve the name with "
    "`entity`, then call `neighborhood`. Example: "
    'search(query="What did the supplier agree to '
    'deliver?")')

_ENTITY_DESC = (
    "Resolve a free-text name to the canonical entity "
    "in the graph: its stable entity_id, aliases, and "
    "how many triples endpoint it. Use this before "
    "`neighborhood` or `connect` whenever you only "
    "have a name. If the name is ambiguous, the result "
    "lists candidates with distinguishers instead of "
    "guessing — pick the matching entity_id. A result "
    "with resolved=false and no candidates means the "
    "corpus holds no such entity; say so rather than "
    "guessing.")

_NEIGHBORHOOD_DESC = (
    "List one entity's relations as triples, each "
    "bound to its evidence sentences — the step a "
    "plain text search cannot do. Requires an "
    "entity_id from `entity` (names are not accepted "
    "here). Optional relation filter narrows to one "
    "relation type. depth=1 lists the entity's own "
    "relations; depth=2 also lists its neighbours'. "
    "Use `triple` on any returned triple_id to audit "
    "its full evidence before relying on it.")

_CONNECT_DESC = (
    "Find the shortest evidence-backed paths between "
    "two entities, for multi-hop questions (\"how is "
    "A connected to B?\"). Requires both entity_ids "
    "from `entity`. Each path is a chain of evidenced "
    "triples, not a claim that the endpoints are "
    "directly related. An empty result means no path "
    "within max_hops exists in this graph. If you "
    "only need one side's relations, `neighborhood` "
    "is cheaper.")

_TRIPLE_DESC = (
    "Audit one triple by its triple_id (from "
    "`search`, `neighborhood`, or `connect`): the "
    "full evidence set — every attested quote with "
    "its source document — plus the provenance chain "
    "and any conflicting triples. Use this before "
    "stating a consequential claim, to verify the "
    "evidence actually says what the triple asserts.")


def build_server(service: McpService):
    """The FastMCP server with the five read-only
    tools bound to ``service``. Import of the SDK
    happens here, so the rest of the module (and its
    tests) never needs the extra installed."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("ontosmith")

    @server.tool(description=_SEARCH_DESC)
    def search(query: str, k: int = 8,
               max_tokens: int = 2000,
               cursor: str | None = None) -> dict:
        return service.search(
            query, k=k, max_tokens=max_tokens,
            cursor=cursor)

    @server.tool(description=_ENTITY_DESC)
    def entity(name: str) -> dict:
        return service.entity(name)

    @server.tool(description=_NEIGHBORHOOD_DESC)
    def neighborhood(entity_id: str,
                     relation: str | None = None,
                     depth: int = 1, limit: int = 20,
                     max_tokens: int = 2000,
                     cursor: str | None = None) -> dict:
        return service.neighborhood(
            entity_id, relation=relation, depth=depth,
            limit=limit, max_tokens=max_tokens,
            cursor=cursor)

    @server.tool(description=_CONNECT_DESC)
    def connect(from_id: str, to_id: str,
                max_hops: int = 3, limit: int = 5,
                max_tokens: int = 2000) -> dict:
        return service.connect(
            from_id, to_id, max_hops=max_hops,
            limit=limit, max_tokens=max_tokens)

    @server.tool(description=_TRIPLE_DESC)
    def triple(triple_id: str) -> dict:
        return service.triple(triple_id)

    return server


def main(argv=None) -> int:
    """``ontosmith mcp [--source PATH] [--batch ID]
    [--store auto|file|neo4j] [--embed]`` — serve the
    built graph to AI callers over MCP (stdio). The
    store binding is the serve surface's: with no
    source, the shipped example build."""
    ap = argparse.ArgumentParser(
        prog="ontosmith mcp",
        description="MCP server over the built graph: "
                    "search / entity / neighborhood / "
                    "connect / triple (read-only, "
                    "evidence-bound)")
    ap.add_argument("--source", default=None,
                    help="batch directory, records "
                         "directory, or records file "
                         "(default: the shipped example "
                         "build)")
    ap.add_argument("--batch", default=None,
                    help="batch id under the state root's "
                         "batches/ (alternative to "
                         "--source)")
    ap.add_argument("--store", default="auto",
                    choices=("auto", "file", "neo4j"),
                    help="serving store: auto = Neo4j "
                         "when NEO4J_URL is configured, "
                         "else the file store "
                         "(default: auto)")
    ap.add_argument("--embed", action="store_true",
                    help="activate the search vector "
                         "arm via the connection layer's "
                         "default embedder")
    args = ap.parse_args(argv)
    # The SDK check comes after parsing, so --help
    # works (and teaches the install line) even when
    # the extra is absent.
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("mcp refused: the MCP SDK is an optional "
              "extra — install it with: "
              'pip install "ontosmith[mcp]"',
              file=sys.stderr)
        return 2
    from pipeline import connections as _connections
    try:
        store = _search.open_store(
            source=args.source, batch=args.batch,
            store=args.store)
        if args.embed:
            from pipeline.embedder import default_embedder
            embedder = default_embedder()
        else:
            embedder = _search.try_embedder()
    except (ValueError, _connections.BindingKeyError) as e:
        print(f"mcp refused: {e}", file=sys.stderr)
        return 1
    service = McpService(store, embedder=embedder)
    desc = store.describe()
    print(f"ontosmith mcp: {desc['kind']} store "
          f"({desc['triples']} triples, "
          f"{desc['documents']} documents) on stdio — "
          "search / entity / neighborhood / connect / "
          "triple", file=sys.stderr)
    build_server(service).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
