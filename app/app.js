"use strict";
/* OntoSmith demo app (plan §7 R18) — a thin client over
   the sample API. It calls exactly the five JSON
   endpoints, with exactly these parameters (pinned by
   tests/test_app.py against api.server.API_ROUTES):
     /health
     /search?q=...&k=10
     /entity?id=...  |  /entity?name=...
     /neighborhood?id=...&hops=1..4
     /triple?subject=...&relation=...&object=...
   No other network calls, no external assets: the app
   works offline against whatever `ontosmith serve` is
   serving. All server text is rendered via textContent
   — the graph's words are data, never markup. */

const $ = (sel) => document.querySelector(sel);
const enc = encodeURIComponent;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function getJson(url) {
  const resp = await fetch(url);
  const body = await resp.json();
  if (!resp.ok) {
    const err = body && body.error
      ? `${body.error.code}: ${body.error.message}`
      : `HTTP ${resp.status}`;
    throw new Error(err);
  }
  return body;
}

function errorBox(err) {
  return el("p", "error-box", `The API refused: ${err.message}`);
}

let storeKind = null;   // "file" | "neo4j" | ... (from /health)
let currentEntityId = null;

/* ------------------------------------------------ health / overview */

function browserUrl(neo4jUrl) {
  /* The Neo4j Browser sits on the same host as the Bolt
     endpoint (7474 next to 7687). The URL the store
     reports carries no credentials — the password is a
     separate binding and never reaches this page. */
  try {
    const u = new URL(neo4jUrl);
    u.protocol = neo4jUrl.includes("+s") ? "https:" : "http:";
    u.port = "7474";
    u.pathname = "/browser/";
    u.search = "";
    u.hash = "";
    return u.toString();
  } catch {
    return null;
  }
}

function renderExplore(store) {
  const block = $("#explore-block");
  block.replaceChildren();
  if (store.kind !== "neo4j") return;
  const url = browserUrl(store.neo4j_url || "");
  if (!url) return;
  block.append(el("h3", null, "Graph exploration"));
  const p = el("p");
  p.append("This graph is served from Neo4j. Rendered graph "
    + "exploration lives in Neo4j Browser, on the same Bolt "
    + "endpoint: ");
  const a = el("a", null, "open Neo4j Browser");
  a.href = url;
  a.target = "_blank";
  a.rel = "noopener";
  p.append(a);
  block.append(p);
  block.append(el("code", "explore-cypher",
    "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 25"));
  block.append(el("p", "muted",
    "An embedded graph explorer is part of the fuller app "
    + "(v1.1); the neighborhood walker in the entity report "
    + "below works over this API for any store."));
}

function renderOverview(store) {
  const body = $("#overview-body");
  body.replaceChildren();
  const dl = el("dl", "facts");
  const rows = [
    ["store", store.kind],
    ["source", store.source || "(configured Neo4j)"],
    ["batch", store.batch_id || "—"],
    ["entities", String(store.entities)],
    ["triples", String(store.triples)],
    ["documents", String(store.documents)],
    ["embeddings", store.embeddings === null
      ? "unknown" : (store.embeddings ? "yes" : "no")],
  ];
  for (const [k, v] of rows) {
    dl.append(el("dt", null, k), el("dd", null, v));
  }
  body.append(dl);
  $("#store-status").textContent =
    `${store.kind} store · ${store.triples} triples · ` +
    `${store.documents} documents`;
}

/* ------------------------------------------------------------- search */

function tripleMeta(t) {
  const p = t.p_true === null || t.p_true === undefined
    ? "p —" : `p ${t.p_true}`;
  return `${p} · tier ${t.tier} · ${t.doc_id || "no doc"}`;
}

function tripleRow(t, onOpen) {
  const b = el("button", "row");
  b.type = "button";
  b.append(el("span", null,
    `${t.subject} — ${t.relation} → ${t.object}`));
  b.append(el("span", "meta", tripleMeta(t)));
  if (t.quote) b.append(el("span", "meta", `“${t.quote}”`));
  b.addEventListener("click", () => onOpen(t));
  return b;
}

async function runSearch(q) {
  const meta = $("#search-meta");
  const results = $("#results");
  results.replaceChildren(el("p", "muted", "Searching…"));
  meta.hidden = true;
  let r;
  try {
    r = await getJson(`/search?q=${enc(q)}&k=10`);
  } catch (err) {
    results.replaceChildren(errorBox(err));
    return;
  }
  /* The arms line is the product's honesty contract:
     name what ran, and print every note verbatim. */
  meta.replaceChildren();
  meta.append(el("span", null, "arms: "));
  meta.append(el("span", "arms",
    r.arms.length ? r.arms.join(" · ") : "(none)"));
  meta.hidden = false;
  results.replaceChildren();
  if (r.notes && r.notes.length) {
    const notes = el("div", "notes");
    for (const n of r.notes) notes.append(el("p", null, n));
    results.append(notes);
  }

  const entBlock = el("div", "result-block");
  entBlock.append(el("h3", null, "Entities"));
  if (r.entities.length) {
    const chips = el("div", "entity-chips");
    for (const e of r.entities) {
      const c = el("button", "chip", e.name);
      c.type = "button";
      c.addEventListener("click",
        () => openEntity({ id: e.entity_id }));
      chips.append(c);
    }
    entBlock.append(chips);
  } else {
    entBlock.append(el("p", "empty",
      "No entity in the question links to the graph."));
  }
  results.append(entBlock);

  const docBlock = el("div", "result-block");
  docBlock.append(el("h3", null, "Documents"));
  if (r.documents.length) {
    for (const d of r.documents) {
      const card = el("div", "doc");
      card.append(el("span", null, d.doc_id));
      card.append(el("span", "meta",
        `score ${Number(d.score).toFixed(6)}`));
      const chunk = (d.chunks || [])[0];
      if (chunk) {
        card.append(el("p", "excerpt",
          chunk.text.replace(/\s+/g, " ").slice(0, 220)));
      }
      docBlock.append(card);
    }
  } else {
    docBlock.append(el("p", "empty", "No document matched."));
  }
  results.append(docBlock);

  const triBlock = el("div", "result-block");
  triBlock.append(el("h3", null, "Triples"));
  if (r.triples.length) {
    for (const t of r.triples) {
      triBlock.append(tripleRow(t, openTriple));
    }
  } else {
    triBlock.append(el("p", "empty", "No triple matched."));
  }
  results.append(triBlock);
}

/* ---------------------------------------------------------- inspector */

async function openTriple(t) {
  const body = $("#inspector-body");
  body.replaceChildren(el("p", "muted", "Loading the report…"));
  let r;
  try {
    r = await getJson(
      `/triple?subject=${enc(t.subject)}` +
      `&relation=${enc(t.relation)}&object=${enc(t.object)}`);
  } catch (err) {
    body.replaceChildren(errorBox(err));
    return;
  }
  body.replaceChildren();
  const stmt = el("p", "statement");
  stmt.append(document.createTextNode(r.triple.subject + " "));
  stmt.append(el("span", "rel", `— ${r.triple.relation} → `));
  stmt.append(document.createTextNode(r.triple.object));
  body.append(stmt);
  if (r.triple.surface_relation) {
    body.append(el("p", "muted",
      `surface form: “${r.triple.surface_relation}”`));
  }
  body.append(el("p", "position",
    `triple ${r.position + 1} of ${r.triples} · via ${r.source}`));

  body.append(el("h3", null, "Evidence"));
  if (r.evidence.length) {
    for (const ev of r.evidence) {
      const q = el("blockquote", "evidence");
      q.append(el("span", "evidence-kind", ev.kind));
      q.append(el("div", null, ev.text));
      body.append(q);
    }
  } else {
    body.append(el("p", "empty", "No evidence span recorded."));
  }

  body.append(el("h3", null, "Provenance"));
  const dl = el("dl", "facts");
  for (const [k, v] of Object.entries(r.provenance)) {
    dl.append(el("dt", null, k), el("dd", null, String(v)));
  }
  body.append(dl);

  const gaps = el("div",
    r.gaps.length ? "gaps" : "gaps none");
  if (r.gaps.length) {
    gaps.append(el("p", null,
      "What this record does not carry:"));
    for (const g of r.gaps) gaps.append(el("p", null, `· ${g}`));
  } else {
    gaps.append(el("p", null,
      "Gaps: none — evidence and provenance are recorded."));
  }
  body.append(gaps);

  const nav = el("p");
  const subj = el("button", "linkish", "subject entity");
  subj.type = "button";
  subj.addEventListener("click",
    () => openEntity({ name: r.triple.subject }));
  const obj = el("button", "linkish", "object entity");
  obj.type = "button";
  obj.addEventListener("click",
    () => openEntity({ name: r.triple.object }));
  nav.append(subj, document.createTextNode(" · "), obj);
  body.append(nav);
  $("#inspector").scrollIntoView({ block: "nearest" });
}

/* ------------------------------------------------------------- entity */

function factGroup(title, facts) {
  const block = el("div", "result-block");
  block.append(el("h3", null, title));
  if (!facts.length) {
    block.append(el("p", "empty", "None."));
    return block;
  }
  for (const t of facts) block.append(tripleRow(t, openTriple));
  return block;
}

async function openEntity(params) {
  const body = $("#entity-body");
  body.replaceChildren(el("p", "muted", "Loading the entity…"));
  let e;
  try {
    e = await getJson(params.id
      ? `/entity?id=${enc(params.id)}`
      : `/entity?name=${enc(params.name)}`);
  } catch (err) {
    body.replaceChildren(errorBox(err));
    return;
  }
  currentEntityId = e.entity_id;
  body.replaceChildren();
  body.append(el("p", "statement", e.name));
  const dl = el("dl", "facts");
  dl.append(el("dt", null, "entity_id"),
    el("dd", null, e.entity_id));
  if (e.aliases && e.aliases.length) {
    dl.append(el("dt", null, "aliases"),
      el("dd", null, e.aliases.join(" · ")));
  }
  dl.append(el("dt", null, "facts"),
    el("dd", null,
      `${e.facts_as_subject} as subject · ` +
      `${e.facts_as_object} as object`));
  body.append(dl);
  body.append(factGroup("As subject",
    e.facts.filter((t) => t.subject_id === e.entity_id)));
  body.append(factGroup("As object",
    e.facts.filter((t) => t.object_id === e.entity_id)));

  const walk = el("div", "result-block");
  walk.append(el("h3", null, "Neighborhood"));
  const controls = el("div", "walk-controls");
  controls.append(el("label", null, "hops"));
  const sel = el("select");
  for (const h of [1, 2, 3, 4]) {
    const opt = el("option", null, String(h));
    opt.value = String(h);
    sel.append(opt);
  }
  const walkBody = el("div");
  const go = el("button", null, "Walk");
  go.type = "button";
  const load = () => loadNeighborhood(e, sel.value, walkBody);
  go.addEventListener("click", load);
  controls.append(sel, go);
  walk.append(controls, walkBody);
  body.append(walk);
  load();
  $("#entity-body").scrollIntoView({ block: "nearest" });
}

async function loadNeighborhood(entity, hops, container) {
  container.replaceChildren(el("p", "muted", "Walking…"));
  let r;
  try {
    r = await getJson(
      `/neighborhood?id=${enc(entity.entity_id)}&hops=${hops}`);
  } catch (err) {
    container.replaceChildren(errorBox(err));
    return;
  }
  container.replaceChildren();
  const names = new Map(
    r.nodes.map((n) => [n.entity_id, n.name]));
  const seen = new Set();
  let count = 0;
  for (const edge of r.edges) {
    const mine = edge.from === entity.entity_id;
    const otherId = mine ? edge.to : edge.from;
    if (otherId === entity.entity_id) continue;
    const key = `${otherId}|${edge.type}|${mine}`;
    if (seen.has(key)) continue;
    seen.add(key);
    count += 1;
    const b = el("button", "row");
    b.type = "button";
    b.append(el("span", null,
      `${mine ? "—" : "←"} ${edge.type} ${mine ? "→" : "—"} ` +
      (names.get(otherId) || otherId)));
    b.append(el("span", "meta",
      `tier ${edge.tier} · p ${edge.p_true} · ${edge.doc_id}`));
    b.addEventListener("click",
      () => openEntity({ id: otherId }));
    container.append(b);
  }
  if (!count) {
    container.append(el("p", "empty",
      "No neighbors within the selected hops."));
  }
}

/* ---------------------------------------------------------------- boot */

async function boot() {
  try {
    const h = await getJson(`/health`);
    storeKind = h.store.kind;
    renderOverview(h.store);
    renderExplore(h.store);
  } catch (err) {
    $("#overview-body").replaceChildren(errorBox(err));
    $("#store-status").textContent = "API unreachable";
  }
  const q = $("#q").value.trim();
  if (q) runSearch(q);
}

$("#search-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  const q = $("#q").value.trim();
  if (q) runSearch(q);
});

boot();
