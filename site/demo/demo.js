'use strict';
/* OntoSmith graph explorer: the full verified build (64 CUAD contracts,
 * 15,413 entities, 11,127 facts) on one canvas. Positions are precomputed
 * per document cluster; interaction isolates neighbourhoods out of the
 * dense field instead of cutting the data. */

const $ = s => document.querySelector(s);
const fmt = n => n.toLocaleString('en-US');
const cv = $('#cv'), ctx2d = cv.getContext('2d');
const stage = $('#stage'), insp = $('#insp');

let G = null, EV = null, evPromise = null;
let adj = [], docFacts = [], lower = [];
let W = 0, H = 0, DPR = 1;
const view = { cx: 3300, cy: 2400, s: 0.1 };
let fitS = 0.1;
let sel = { ent: -1, triple: -1, doc: -1 };
let iso = null;               // {kind:'ent', id, depth} | {kind:'doc', id}
let focusEnts = null, focusTris = null, dist = null;
let relSel = new Set();
let hover = -1;
let needDraw = false;

const TOK = {};
function readTokens() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ['--edge', '--edge-faint', '--edge-lit', '--node', '--node-dim',
                   '--node-sel', '--ring', '--caption', '--cream', '--amber-2', '--clay-2'])
    TOK[k] = cs.getPropertyValue(k).trim();
}
const FONT = "11.5px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif";

/* ---------------- data ---------------- */
fetch('graph.json').then(r => r.json()).then(g => {
  G = g;
  adj = Array.from({ length: G.ent.length }, () => []);
  docFacts = Array.from({ length: G.docs.length }, () => []);
  for (let t = 0; t < G.ts.length; t++) {
    adj[G.ts[t]].push(t); adj[G.to[t]].push(t);
    docFacts[G.td[t]].push(t);
  }
  lower = G.ent.map(s => s.toLowerCase());
  buildGrid();
  const m = G._meta;
  for (const k of ['documents', 'entities', 'facts', 'relations']) {
    const el = document.querySelector(`[data-stat="${k}"]`);
    if (el) el.textContent = fmt(m[k]);
  }
  buildDocList(); buildRelList();
  resize();
  view.cx = m.world[0] / 2; view.cy = m.world[1] / 2;
  fitView(false);
  $('#loading').style.display = 'none';
  renderDefault();
  updateStatus();
}).catch(() => { $('#loading').textContent = 'The graph data could not be loaded.'; });

function loadEvidence() {
  if (EV) return Promise.resolve(EV);
  if (!evPromise) evPromise = fetch('evidence.json').then(r => r.json()).then(e => { EV = e; return e; });
  return evPromise;
}

/* ---------------- view / draw ---------------- */
function fitView(clearSel = true) {
  fitS = Math.min(W / G._meta.world[0], H / G._meta.world[1]) * 0.97;
  view.cx = G._meta.world[0] / 2; view.cy = G._meta.world[1] / 2; view.s = fitS;
  if (clearSel) { iso = null; focusEnts = focusTris = null; sel = { ent: -1, triple: -1, doc: -1 }; markDocOn(); renderDefault(); }
  requestDraw(); updateStatus();
}
function requestDraw() { if (!needDraw) { needDraw = true; requestAnimationFrame(() => { needDraw = false; draw(); }); } }
function sx(x) { return (x - view.cx) * view.s + W / 2; }
function sy(y) { return (y - view.cy) * view.s + H / 2; }
function wx(px) { return (px - W / 2) / view.s + view.cx; }
function wy(py) { return (py - H / 2) / view.s + view.cy; }

function draw() {
  if (!G || !W) return;
  ctx2d.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx2d.clearRect(0, 0, W, H);
  const s = view.s;
  const relActive = relSel.size > 0;
  const pass = t => !relActive || relSel.has(G.tr[t]);

  /* cluster rings */
  ctx2d.lineWidth = 1;
  ctx2d.strokeStyle = TOK['--ring'];
  for (let d = 0; d < G.docs.length; d++) {
    const doc = G.docs[d], cxp = sx(doc.x), cyp = sy(doc.y), r = doc.r * s;
    if (cxp + r < -40 || cxp - r > W + 40 || cyp + r < -40 || cyp - r > H + 40) continue;
    ctx2d.beginPath(); ctx2d.arc(cxp, cyp, Math.max(2, r), 0, 6.2832); ctx2d.stroke();
  }

  /* edges, faint pass: alpha thins as the field zooms out, so clusters
     read as nebulae at overview instead of saturating into flat discs */
  ctx2d.lineWidth = 1;
  ctx2d.strokeStyle = TOK['--edge'];
  ctx2d.globalAlpha = iso ? 0.16 : Math.max(0.18, Math.min(1, s * 2.6));
  ctx2d.beginPath();
  for (let t = 0; t < G.ts.length; t++) {
    if (!pass(t) || (focusTris && focusTris.has(t))) continue;
    const a = G.ts[t], b = G.to[t];
    const x1 = sx(G.ex[a]), y1 = sy(G.ey[a]), x2 = sx(G.ex[b]), y2 = sy(G.ey[b]);
    if ((x1 < -30 && x2 < -30) || (x1 > W + 30 && x2 > W + 30) || (y1 < -30 && y2 < -30) || (y1 > H + 30 && y2 > H + 30)) continue;
    ctx2d.moveTo(x1, y1); ctx2d.lineTo(x2, y2);
  }
  ctx2d.stroke();
  ctx2d.globalAlpha = 1;

  /* edges, lit pass: focus set, hover neighbourhood, selected triple */
  const lit = new Set();
  if (focusTris) for (const t of focusTris) if (pass(t)) lit.add(t);
  if (hover >= 0 && !iso) for (const t of adj[hover]) if (pass(t)) lit.add(t);
  if (lit.size) {
    ctx2d.strokeStyle = TOK['--edge-lit']; ctx2d.lineWidth = 1.1; ctx2d.globalAlpha = 0.85;
    ctx2d.beginPath();
    for (const t of lit) {
      const a = G.ts[t], b = G.to[t];
      ctx2d.moveTo(sx(G.ex[a]), sy(G.ey[a])); ctx2d.lineTo(sx(G.ex[b]), sy(G.ey[b]));
    }
    ctx2d.stroke(); ctx2d.globalAlpha = 1;
  }
  if (sel.triple >= 0 && pass(sel.triple)) {
    const a = G.ts[sel.triple], b = G.to[sel.triple];
    ctx2d.strokeStyle = TOK['--clay-2']; ctx2d.lineWidth = 2.2;
    ctx2d.beginPath(); ctx2d.moveTo(sx(G.ex[a]), sy(G.ey[a])); ctx2d.lineTo(sx(G.ex[b]), sy(G.ey[b])); ctx2d.stroke();
  }

  /* nodes */
  ctx2d.globalAlpha = iso ? 1 : Math.max(0.38, Math.min(1, s * 3.2));
  for (let i = 0; i < G.ent.length; i++) {
    const x = sx(G.ex[i]), y = sy(G.ey[i]);
    if (x < -12 || x > W + 12 || y < -12 || y > H + 12) continue;
    let rp = Math.min(6, 1.15 + Math.sqrt(G.edeg[i]) * 0.5);
    if (s < 0.2) rp = Math.min(rp, 0.75 + Math.sqrt(G.edeg[i]) * 0.05);
    let col = TOK['--node'];
    if (iso && focusEnts && !focusEnts.has(i)) col = TOK['--node-dim'];
    if (i === sel.ent) col = TOK['--node-sel'];
    else if (i === hover) col = TOK['--amber-2'];
    ctx2d.fillStyle = col;
    if (rp <= 1.7) ctx2d.fillRect(x - rp, y - rp, rp * 2, rp * 2);
    else { ctx2d.beginPath(); ctx2d.arc(x, y, rp, 0, 6.2832); ctx2d.fill(); }
  }
  ctx2d.globalAlpha = 1;

  /* labels */
  ctx2d.font = FONT;
  const placed = [];
  const tryLabel = (i, text) => {
    const x = sx(G.ex[i]) + 8, y = sy(G.ey[i]) + 4;
    if (x < 0 || x > W - 40 || y < 8 || y > H - 4) return;
    const w = ctx2d.measureText(text).width;
    for (const b of placed) if (x < b[0] + b[2] && x + w > b[0] && y - 9 < b[1] + b[3] && y + 3 > b[1]) return;
    placed.push([x, y - 9, w, 12]);
    ctx2d.fillStyle = 'rgba(10,22,32,.72)';
    ctx2d.fillRect(x - 2, y - 9.5, w + 4, 13);
    ctx2d.fillStyle = TOK['--cream'];
    ctx2d.fillText(text, x, y);
  };
  if (iso && focusEnts) {
    const cand = [...focusEnts].sort((a, b) => G.edeg[b] - G.edeg[a]).slice(0, 110);
    for (const i of cand) tryLabel(i, G.ent[i]);
  } else if (s >= 0.34) {
    for (let i = 0; i < G.ent.length; i++) if (G.edeg[i] >= 60) tryLabel(i, G.ent[i]);
  }
  /* document names over clusters when zoomed out, collision-thinned */
  if (s < 0.34 && !iso) {
    ctx2d.font = FONT; ctx2d.textAlign = 'center';
    const dplaced = [];
    for (let d = 0; d < G.docs.length; d++) {
      const doc = G.docs[d], x = sx(doc.x), y = sy(doc.y), rr = doc.r * s;
      if (rr < 17 || x < 30 || x > W - 30 || y < 10 || y > H) continue;
      const name = doc.c.length > 22 ? doc.c.slice(0, 21) + '…' : doc.c;
      const w = ctx2d.measureText(name).width, ly = y - rr - 6;
      let clash = false;
      for (const b of dplaced) if (x - w / 2 < b[0] + b[2] && x + w / 2 > b[0] && ly - 9 < b[1] + b[3] && ly + 3 > b[1]) { clash = true; break; }
      if (clash) continue;
      dplaced.push([x - w / 2, ly - 9, w, 12]);
      ctx2d.fillStyle = TOK['--caption'];
      ctx2d.fillText(name, x, ly);
    }
    ctx2d.textAlign = 'left';
  }
  updateStatus();
}

function updateStatus() {
  if (!G) return;
  let shown = G.ts.length;
  if (relSel.size) { shown = 0; for (let t = 0; t < G.ts.length; t++) if (relSel.has(G.tr[t])) shown++; }
  $('#st-facts').innerHTML = '';
  const b = document.createElement('b'); b.textContent = fmt(shown);
  $('#st-facts').append(b, document.createTextNode(` of ${fmt(G.ts.length)} facts drawn`));
  $('#st-zoom').textContent = `Zoom ${view.s.toFixed(2)}×`;
}

/* ---------------- focus / selection ---------------- */
function computeFocus() {
  focusEnts = null; focusTris = null; dist = null;
  if (!iso) return;
  if (iso.kind === 'doc') {
    focusTris = new Set(docFacts[iso.id]);
    focusEnts = new Set();
    for (const t of focusTris) { focusEnts.add(G.ts[t]); focusEnts.add(G.to[t]); }
    return;
  }
  dist = new Map([[iso.id, 0]]);
  let frontier = [iso.id];
  for (let d = 1; d <= iso.depth; d++) {
    const next = [];
    for (const e of frontier) for (const t of adj[e]) {
      for (const o of [G.ts[t], G.to[t]]) if (!dist.has(o)) { dist.set(o, d); next.push(o); }
    }
    frontier = next;
  }
  focusEnts = new Set(dist.keys());
  focusTris = new Set();
  for (const [e, d] of dist) {
    if (d >= iso.depth) continue;
    for (const t of adj[e]) if (focusEnts.has(G.ts[t]) && focusEnts.has(G.to[t])) focusTris.add(t);
  }
}

function fitFocus() {
  if (!focusEnts || !focusEnts.size) return;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const e of focusEnts) {
    x0 = Math.min(x0, G.ex[e]); x1 = Math.max(x1, G.ex[e]);
    y0 = Math.min(y0, G.ey[e]); y1 = Math.max(y1, G.ey[e]);
  }
  const bw = Math.max(60, x1 - x0), bh = Math.max(60, y1 - y0);
  view.s = Math.max(fitS * 0.9, Math.min(1.15, Math.min(W / bw, H / bh) * 0.86));
  view.cx = (x0 + x1) / 2; view.cy = (y0 + y1) / 2;
}
function selectEntity(i, centre = false) {
  sel.ent = i; sel.triple = -1;
  iso = { kind: 'ent', id: i, depth: iso && iso.kind === 'ent' && iso.id === i ? iso.depth : 1 };
  sel.doc = G.edoc[i];
  computeFocus();
  if (centre) fitFocus();
  markDocOn(); renderEntity(); requestDraw();
}
function selectDoc(d, toggle = true) {
  if (toggle && iso && iso.kind === 'doc' && iso.id === d) { clearSelection(); return; }
  sel.doc = d; sel.ent = -1; sel.triple = -1;
  iso = { kind: 'doc', id: d };
  computeFocus();
  const doc = G.docs[d];
  view.cx = doc.x; view.cy = doc.y;
  view.s = Math.max(fitS, Math.min(1.15, Math.min(W, H) / (2.5 * doc.r)));
  markDocOn(); renderDoc(); requestDraw();
}
function clearSelection() {
  sel = { ent: -1, triple: -1, doc: -1 }; iso = null; focusEnts = focusTris = null;
  markDocOn(); renderDefault(); requestDraw();
}
function markDocOn() {
  document.querySelectorAll('.docitem').forEach(el =>
    el.classList.toggle('on', iso && iso.kind === 'doc' && iso.id === +el.dataset.doc));
}

/* ---------------- inspector ---------------- */
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function backBtn(label, fn) {
  const b = el('button', 'backlink', label); b.type = 'button'; b.addEventListener('click', fn); return b;
}
function renderDefault() {
  insp.replaceChildren();
  insp.append(el('h3', '', 'Inspector'));
  const p = el('p', 'empty', 'Click any node in the field and the inspector lists its facts. Pick a fact and it shows the evidence: the sentence from the contract, in context. The document index isolates one contract at a time; the relation forms thin the field live.');
  p.style.marginTop = '8px'; insp.append(p);
  insp.append(el('h2', '', 'Largest entities'));
  const top = G.ent.map((_, i) => i).sort((a, b) => G.edeg[b] - G.edeg[a]).slice(0, 8);
  const list = el('div', 'factlist');
  for (const i of top) {
    const b = el('button', 'fact'); b.type = 'button';
    b.append(el('span', 'o', G.ent[i]), el('span', 'd', `${fmt(G.edeg[i])} facts`));
    b.addEventListener('click', () => selectEntity(i, true));
    list.append(b);
  }
  insp.append(list);
}
function factRow(t, anchor) {
  const out = G.ts[t] === anchor;
  const other = out ? G.to[t] : G.ts[t];
  const b = el('button', 'fact' + (t === sel.triple ? ' on' : '')); b.type = 'button';
  const r = el('span', 'r', G.rels[G.tr[t]]);
  const dir = el('span', 'dir', out ? '  outgoing' : '  incoming');
  r.append(dir);
  b.append(r, el('span', 'o', G.ent[other]),
           el('span', 'd', `${G.docs[G.td[t]].c}: ${G.docs[G.td[t]].t}`));
  b.addEventListener('click', () => { sel.triple = t; renderEvidence(t); requestDraw(); });
  return b;
}
function renderEntity() {
  const i = sel.ent;
  insp.replaceChildren();
  insp.append(backBtn('Back to overview', clearSelection));
  insp.append(el('h3', '', G.ent[i]));
  const meta = el('div', 'meta');
  const home = G.docs[G.edoc[i]];
  meta.append(el('div', '', `Home document: ${home.t} (${home.c})`));
  meta.append(el('div', '', `${fmt(G.edeg[i])} facts`));
  if (G.endocs[i] > 1) meta.append(el('div', '', `Appears in ${G.endocs[i]} documents`));
  insp.append(meta);
  const row = el('div', 'row');
  for (const d of [1, 2]) {
    const b = el('button', 'btn' + (iso.depth === d ? ' primary' : ''), `${d} hop${d > 1 ? 's' : ''}`);
    b.type = 'button';
    b.addEventListener('click', () => { iso.depth = d; computeFocus(); fitFocus(); renderEntity(); requestDraw(); });
    row.append(b);
  }
  const c = el('button', 'btn', 'Centre in graph'); c.type = 'button';
  c.addEventListener('click', () => { fitFocus(); requestDraw(); });
  row.append(c);
  insp.append(row);
  let tris = adj[i].slice();
  if (relSel.size) tris = tris.filter(t => relSel.has(G.tr[t]));
  tris.sort((a, b) => G.rels[G.tr[a]].localeCompare(G.rels[G.tr[b]]));
  insp.append(el('h2', '', relSel.size ? `Facts (${tris.length} after relation filter)` : `Facts (${tris.length})`));
  const list = el('div', 'factlist scroll');
  for (const t of tris) list.append(factRow(t, i));
  insp.append(list);
}
function renderEvidence(t) {
  const s = G.ts[t], o = G.to[t], d = G.td[t];
  insp.replaceChildren();
  insp.append(backBtn(sel.ent >= 0 ? 'Back to the entity' : 'Back to overview',
    () => { if (sel.ent >= 0) renderEntity(); else renderDefault(); }));
  insp.append(el('h2', '', 'Fact and evidence'));
  const f = el('p', 'evfact');
  f.append(document.createTextNode(G.ent[s] + ' '), el('span', 'r', G.rels[G.tr[t]]),
           document.createTextNode(' ' + G.ent[o]));
  insp.append(f);
  const row = el('div', 'row');
  const bs = el('button', 'btn', 'Subject entity'); bs.type = 'button';
  bs.addEventListener('click', () => selectEntity(s, true));
  const bo = el('button', 'btn', 'Object entity'); bo.type = 'button';
  bo.addEventListener('click', () => selectEntity(o, true));
  row.append(bs, bo); insp.append(row);
  const bq = el('blockquote', 'ev', 'Loading the evidence…');
  insp.append(bq);
  const src = el('div', 'src');
  src.append(el('div', '', `Source: ${G.docs[d].t} (${G.docs[d].c})`), el('div', 'fn', G.docs[d].f));
  insp.append(src);
  loadEvidence().then(() => {
    const pair = EV.tq[t], q = EV.q[pair], cx = EV.ctx[pair];
    bq.replaceChildren();
    if (cx) {
      bq.append(el('span', 'ctx', '…' + cx[0]));
      const mk = el('mark', '', q); bq.append(mk);
      bq.append(el('span', 'ctx', cx[1] + '…'));
    } else bq.append(el('span', '', q));
  }).catch(() => { bq.textContent = 'The evidence file could not be loaded.'; });
}
function renderDoc() {
  const d = sel.doc, doc = G.docs[d];
  insp.replaceChildren();
  insp.append(backBtn('Back to overview', clearSelection));
  insp.append(el('h3', '', doc.t));
  const meta = el('div', 'meta');
  meta.append(el('div', '', doc.c));
  meta.append(el('div', '', `${fmt(doc.n)} facts, ${fmt(doc.e)} entities homed here`));
  insp.append(meta);
  const fn = el('div', 'src'); fn.append(el('div', 'fn', doc.f)); insp.append(fn);
  const local = new Map();
  for (const t of docFacts[d]) for (const e of [G.ts[t], G.to[t]]) local.set(e, (local.get(e) || 0) + 1);
  const top = [...local.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
  insp.append(el('h2', '', 'Busiest entities in this contract'));
  const list = el('div', 'factlist');
  for (const [e, n] of top) {
    const b = el('button', 'fact'); b.type = 'button';
    b.append(el('span', 'o', G.ent[e]), el('span', 'd', `${n} facts in this document`));
    b.addEventListener('click', () => selectEntity(e, true));
    list.append(b);
  }
  insp.append(list);
}

/* ---------------- left rail: documents + relations ---------------- */
function buildDocList() {
  const box = $('#doclist');
  G.docs.forEach((doc, d) => {
    const b = el('button', 'docitem'); b.type = 'button'; b.dataset.doc = d;
    b.append(el('span', 't', doc.t), el('span', 'm', `${doc.c} — ${fmt(doc.n)} facts`));
    b.addEventListener('click', () => selectDoc(d));
    box.append(b);
  });
}
function buildRelList(filter = '') {
  const box = $('#rellist'); box.replaceChildren();
  const q = filter.toLowerCase();
  let shown = 0;
  for (let r = 0; r < G.rels.length && shown < 60; r++) {
    if (q && !G.rels[r].toLowerCase().includes(q)) continue;
    shown++;
    const lab = el('label', 'relitem');
    const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = relSel.has(r);
    cb.addEventListener('change', () => {
      if (cb.checked) relSel.add(r); else relSel.delete(r);
      if (sel.ent >= 0 && sel.triple < 0) renderEntity();
      updateStatus(); requestDraw();
    });
    lab.append(cb, el('span', 'rn', G.rels[r]), el('span', 'rc', fmt(G.relN[r])));
    box.append(lab);
  }
  if (!shown) box.append(el('p', 'empty', 'No relation form matches.'));
}
$('#relq').addEventListener('input', e => { if (G) buildRelList(e.target.value); });
$('#relclear').addEventListener('click', () => {
  relSel.clear();
  document.querySelectorAll('#rellist input[type=checkbox]').forEach(cb => { cb.checked = false; });
  if (sel.ent >= 0 && sel.triple < 0) renderEntity();
  updateStatus(); requestDraw();
});

/* ---------------- search ---------------- */
const qInput = $('#q'), results = $('#results');
function closeResults() { results.classList.remove('open'); results.replaceChildren(); }
qInput.addEventListener('input', () => {
  const q = qInput.value.trim().toLowerCase();
  if (!G || q.length < 2) { closeResults(); return; }
  const hits = [];
  for (let i = 0; i < lower.length && hits.length < 400; i++) if (lower[i].includes(q)) hits.push(i);
  hits.sort((a, b) => G.edeg[b] - G.edeg[a]);
  results.replaceChildren();
  for (const i of hits.slice(0, 10)) {
    const b = el('button', ''); b.type = 'button'; b.setAttribute('role', 'option');
    b.append(el('span', '', G.ent[i]), el('span', 'deg', `${fmt(G.edeg[i])} facts`));
    b.addEventListener('click', () => { qInput.value = G.ent[i]; closeResults(); selectEntity(i, true); });
    results.append(b);
  }
  results.classList.toggle('open', hits.length > 0);
});
qInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && results.firstChild) results.firstChild.click();
  if (e.key === 'Escape') closeResults();
});
document.addEventListener('pointerdown', e => { if (!e.target.closest('.searchbox')) closeResults(); });

/* ---------------- canvas interaction ---------------- */
const grid = new Map();
function buildGrid() {
  const cell = 64;
  for (let i = 0; i < G.ent.length; i++) {
    const k = ((G.ex[i] / cell) | 0) + ':' + ((G.ey[i] / cell) | 0);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
}
function hitNode(px, py) {
  const x = wx(px), y = wy(py), cell = 64;
  const rad = Math.max(11 / view.s, 12);
  let best = -1, bd = rad * rad;
  const gx = (x / cell) | 0, gy = (y / cell) | 0;
  for (let a = gx - 1; a <= gx + 1; a++) for (let b = gy - 1; b <= gy + 1; b++) {
    const arr = grid.get(a + ':' + b); if (!arr) continue;
    for (const i of arr) {
      const dx = G.ex[i] - x, dy = G.ey[i] - y, d2 = dx * dx + dy * dy;
      if (d2 < bd) { bd = d2; best = i; }
    }
  }
  return best;
}
function segDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy || 1;
  let t = ((px - x1) * dx + (py - y1) * dy) / L; t = Math.max(0, Math.min(1, t));
  const ex = x1 + t * dx - px, ey = y1 + t * dy - py;
  return Math.hypot(ex, ey);
}
let drag = null;
cv.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy, moved: false };
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', e => {
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  if (drag) {
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; cv.classList.add('drag'); }
    if (drag.moved) { view.cx = drag.cx - dx / view.s; view.cy = drag.cy - dy / view.s; requestDraw(); }
    return;
  }
  if (!G) return;
  const h = hitNode(px, py);
  if (h !== hover) { hover = h; cv.style.cursor = h >= 0 ? 'pointer' : 'grab'; requestDraw(); }
});
cv.addEventListener('pointerup', e => {
  const wasDrag = drag && drag.moved; drag = null; cv.classList.remove('drag');
  if (wasDrag || !G) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const n = hitNode(px, py);
  if (n >= 0) { selectEntity(n); return; }
  if (iso && focusTris) {
    for (const t of focusTris) {
      if (relSel.size && !relSel.has(G.tr[t])) continue;
      if (segDist(px, py, sx(G.ex[G.ts[t]]), sy(G.ey[G.ts[t]]), sx(G.ex[G.to[t]]), sy(G.ey[G.to[t]])) < 6) {
        sel.ent = -1; sel.triple = t; renderEvidence(t); requestDraw(); return;
      }
    }
  }
  if (iso || sel.ent >= 0) clearSelection();
});
cv.addEventListener('wheel', e => {
  e.preventDefault();
  if (!G) return;
  const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const wxp = wx(px), wyp = wy(py);
  const ns = Math.max(fitS * 0.45, Math.min(7, view.s * Math.exp(-e.deltaY * 0.0013)));
  view.s = ns;
  view.cx = wxp - (px - W / 2) / ns; view.cy = wyp - (py - H / 2) / ns;
  requestDraw();
}, { passive: false });
cv.addEventListener('keydown', e => {
  const step = 90 / view.s;
  if (e.key === 'ArrowLeft') view.cx -= step;
  else if (e.key === 'ArrowRight') view.cx += step;
  else if (e.key === 'ArrowUp') view.cy -= step;
  else if (e.key === 'ArrowDown') view.cy += step;
  else if (e.key === '+' || e.key === '=') view.s = Math.min(7, view.s * 1.2);
  else if (e.key === '-') view.s = Math.max(fitS * 0.45, view.s / 1.2);
  else return;
  e.preventDefault(); requestDraw();
});
$('#fit').addEventListener('click', () => fitView(true));

/* ---------------- sizing ---------------- */
function resize() {
  const r = stage.getBoundingClientRect();
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = Math.max(280, r.width); H = Math.max(320, r.height);
  cv.width = W * DPR; cv.height = H * DPR;
  draw();   // synchronous: a resize clears the bitmap, never leave it blank
}
new ResizeObserver(resize).observe(stage);

readTokens();
