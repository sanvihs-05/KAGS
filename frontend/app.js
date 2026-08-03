/* ============================================================
   FBSL-KAGS frontend logic
   - Live mode: POST /pipeline/run (full multi-agent pipeline)
   - Sample mode: fetch sample_result.json (a captured real run)
   Renders top outputs + the real Graph-of-Thoughts prune/aggregate trace.
   ============================================================ */
const API_URL = (window.API_URL || '/pipeline/run');

const $ = (id) => document.getElementById(id);
const form = $('pipeline-form');
const statusEl = $('status');

const DIMS = [
  ['functional_adequacy', 'Functional'],
  ['behavioral_performance', 'Behavioral'],
  ['structural_feasibility', 'Structural'],
  ['layout_efficiency', 'Layout'],
  ['sustainability', 'Sustainability'],
];

/* ── helpers ─────────────────────────────────────────────── */
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const pct = (v) => `${(Number(v) * 100).toFixed(1)}%`;
const f3 = (v) => Number(v ?? 0).toFixed(3);

function prettyVariant(v) {
  if (!v || v === 'N/A') return 'Design variant';
  if (v === 'aggregated_hybrid') return 'Aggregated hybrid';
  return String(v).split('+')
    .map((seg) => {
      const s = seg.replace(/_/g, ' ').trim();
      return s.charAt(0).toUpperCase() + s.slice(1);
    })
    .join(' + ');
}

function setStatus(msg, kind) {
  statusEl.hidden = false;
  statusEl.className = 'status' + (kind === 'error' ? ' error' : '');
  const spin = kind === 'busy' ? '<span class="spinner"></span>' : '';
  statusEl.innerHTML = spin + `<span>${esc(msg)}</span>`;
}
function hideStatus() { statusEl.hidden = true; }

/* ── example briefs (low- vs high-complexity, real tested scenarios) ── */
const EXAMPLES = {
  simple: {
    name: 'Simple 2-Bedroom Apartment',
    brief: 'A small 2-bedroom apartment with one bathroom, a kitchen, and a living room. Around 70 sqm.',
  },
  complex: {
    name: '4-Bedroom Family Home',
    brief: 'Design a 4-bedroom family home of 220–260 sqm. The master bedroom should be 18 sqm '
      + 'with an ensuite bathroom, and three further bedrooms of 12–14 sqm each sharing a common '
      + 'bathroom. Provide an open-plan kitchen and living room of about 40 sqm, with the kitchen '
      + 'connected to a separate dining area of 12 sqm. Include a quiet home office of 10 sqm, a '
      + 'sauna, a laundry, and a mudroom that connects to a garage. Prioritise natural light '
      + 'throughout and good acoustic separation between the bedrooms and living spaces.',
  },
};
document.querySelectorAll('.ex-chip').forEach((btn) => {
  btn.addEventListener('click', () => {
    const ex = EXAMPLES[btn.dataset.ex];
    if (!ex) return;
    $('requirements').value = ex.brief;
    $('project_name').value = ex.name;
    $('requirements').focus();
  });
});

/* ── theme ───────────────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem('kags-theme');
  if (saved) document.documentElement.setAttribute('data-theme', saved);
  $('theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const isDark = cur === 'dark' ||
      (!cur && window.matchMedia('(prefers-color-scheme: dark)').matches);
    const next = isDark ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('kags-theme', next);
  });
})();

/* ── render: summary tiles ───────────────────────────────── */
function renderSummary(data) {
  const el = $('summary');
  const gg = data.got_graph;
  const cx = data.complexity_metrics || {};
  const tiles = [];
  const tile = (k, v, u) => tiles.push(
    `<div class="tile"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>${u ? `<div class="u">${esc(u)}</div>` : ''}</div>`);

  tile('Method', data.method === 'Graph of Thought' ? 'Graph of Thought' : 'Traditional');
  tile('Designs returned', (data.designs || []).length, 'after prune + hybrid');
  if (gg) {
    const dropped = gg.prune.n_scored - gg.prune.n_kept;
    tile('Candidates', gg.prune.n_scored, `${gg.prune.n_kept} kept · ${dropped} dropped`);
    tile('Graph', gg.graph_statistics.total_nodes, `${gg.graph_statistics.total_edges} edges · depth ${gg.graph_statistics.graph_depth}`);
  }
  if (cx.level) tile('Complexity', cx.level.charAt(0).toUpperCase() + cx.level.slice(1),
    `${cx.room_count} rooms · ${cx.function_count} functions`);
  if (data.processing_time != null)
    tile('Runtime', `${Number(data.processing_time).toFixed(1)}s`, 'wall clock');

  el.innerHTML = tiles.join('');
  el.hidden = false;
}

/* ── render: one design card ─────────────────────────────── */
function renderDesign(d, rank) {
  const s = d.scores || {};
  const isHybrid = d.variant_type === 'aggregated_hybrid';

  const meters = DIMS.map(([key, label]) => {
    const v = Number(s[key] ?? 0);
    return `<div class="meter">
      <span class="m-label">${label}</span>
      <div class="m-track"><div class="m-fill" style="width:${Math.max(0, Math.min(100, v * 100))}%"></div></div>
      <span class="m-val">${v.toFixed(2)}</span>
    </div>`;
  }).join('');

  // Prefer PNG (raster) prototypes; fall back to inline SVG.
  const fpPng = d.png_floor_plan, adjPng = d.png_adjacency;
  const fp = fpPng
    ? `<img class="plan-img" alt="Floor plan" src="${fpPng}">`
    : (d.svg_floor_plan || '');
  const adj = adjPng
    ? `<img class="plan-img" alt="Connectivity graph" src="${adjPng}">`
    : (d.adjacency_svg || '');
  const hasPlan = !!fp;
  const hasAdj = !!adj;

  const chips = [
    `<span class="chip">${d.functions_count ?? 0} functions</span>`,
    `<span class="chip">${d.behaviors_count ?? 0} behaviors</span>`,
    `<span class="chip">${d.structures_count ?? 0} structures</span>`,
    d.converged ? `<span class="chip good">✓ converged (${d.convergence_iterations})</span>`
                : `<span class="chip">${d.convergence_iterations ?? 0} iterations</span>`,
  ].join('');

  return `<article class="design rank-${rank}">
    <div class="design-head">
      <div class="rank-badge">${rank}</div>
      <div class="t">
        <div class="variant">${esc(prettyVariant(d.variant_type))}</div>
        ${isHybrid ? '<span class="hybrid-tag">◇ merged from top variants</span>' : ''}
        ${d.description && d.description !== 'N/A' ? `<div class="desc">${esc(d.description)}</div>` : ''}
      </div>
      <div class="composite">
        <div class="num">${f3(s.composite)}</div>
        <div class="lbl">composite</div>
      </div>
      ${d.prototype_id ? `<button type="button" class="icon-btn danger card-del"
        data-proto="${esc(d.prototype_id)}" title="Delete this prototype">✕</button>` : ''}
    </div>
    <div class="meters">${meters}</div>
    ${(hasPlan || hasAdj) ? `
      <div class="plan-view">
        ${hasPlan ? `<div class="plan-block"><div class="plan-cap">Floor plan</div><div class="svg-wrap">${fp}</div></div>` : ''}
        ${hasAdj ? `<div class="plan-block"><div class="plan-cap">Connectivity graph</div><div class="svg-wrap">${adj}</div></div>` : ''}
      </div>` : '<div class="plan-view"><div class="svg-wrap empty">No layout generated</div></div>'}
    <div class="chips">${chips}</div>
  </article>`;
}

function renderDesigns(data, topK) {
  const wrap = $('designs');
  const designs = (data.designs || []).slice(0, topK);
  wrap.innerHTML = designs.map((d, i) => renderDesign(d, i + 1)).join('');
  $('designs-section').hidden = designs.length === 0;

  // Per-prototype delete (only present for prototypes loaded from the store)
  wrap.querySelectorAll('.card-del').forEach((btn) => {
    btn.addEventListener('click', () =>
      deletePrototype(btn.dataset.proto, btn.closest('.design')));
  });
}

/* ── render: GoT prune chart ─────────────────────────────── */
function renderPrune(gg) {
  const p = gg.prune;
  const sourceIds = new Set((gg.aggregation.sources || []).map((s) => s.id));

  // Two real cut mechanisms: (1) score below the 0.70×best threshold,
  // (2) above threshold but dropped by the diversity / top-N cap.
  const rows = gg.candidates.map((c) => {
    const belowThr = c.score < p.threshold;
    const reason = c.brief_error ? 'violation'
      : (c.kept ? 'kept' : (belowThr ? 'threshold' : 'cap'));
    const cls = [c.kept ? 'kept' : 'pruned'];
    if (c.kept && sourceIds.has(c.id)) cls.push('source');
    const w = Math.max(0, Math.min(100, c.score * 100));
    const tag = {
      kept: ['k', 'KEPT'], threshold: ['p', '&lt; thr'],
      cap: ['c', 'cap'], violation: ['p', 'brief'],
    }[reason];
    const label = prettyVariant(c.variant_type)
      + (c.brief_error ? ` — ${c.brief_error}` : '');
    return `<div class="prow ${cls.join(' ')} r-${reason}">
      <span class="pv" title="${esc(label)}">${esc(label)}</span>
      <div class="pbar-wrap">
        <div class="pbar-track"></div>
        <div class="pbar-fill" style="width:${w}%"></div>
      </div>
      <span class="ptag"><span class="sc">${c.score.toFixed(3)}</span><span class="st ${tag[0]}">${tag[1]}</span></span>
    </div>`;
  }).join('');

  // No candidate could satisfy the brief (typically the room program's minimum
  // areas exceed the stated total). Ranking falls back to ungated scores, so the
  // chart below is meaningful — but every design still violates the brief.
  const warn = p.gate_fallback ? `<div class="gate-warn">
      <strong>No design satisfies the brief.</strong> Every candidate violated it —
      most often the requested room program cannot fit the stated total area — so
      the brief gate could not separate them. Ranking below falls back to the
      ungated scores; treat these as the best available compromises, not compliant
      designs. Each row shows its specific violation.
    </div>` : '';

  return `<div class="got-panel card">
    <h3>Prune — ${p.n_scored} candidates scored → ${p.n_kept} kept</h3>
    ${warn}
    <p class="p-sub">Two cuts. First the <strong>0.70 × best</strong> threshold (${p.top_score.toFixed(3)} → <strong>${p.threshold.toFixed(3)}</strong>, the orange line) drops weak and brief-violating designs — the latter forced to 0.000. Then a diversity cap keeps the top ${p.n_kept} distinct, so some designs right of the line (grey, “cap”) are still cut.</p>
    <div class="prune-legend">
      <span class="lg"><span class="sw kept"></span>kept</span>
      <span class="lg"><span class="sw pruned"></span>cut (below threshold / capped)</span>
      <span class="lg"><span class="sw thr"></span>prune threshold</span>
      <span class="lg"><span class="sw" style="background:transparent;box-shadow:inset 0 0 0 2px var(--source);"></span>aggregation source</span>
    </div>
    <div class="prune-rows">
      ${rows}
      <div class="pbar-thr-overlay"></div>
    </div>
    <p class="prune-note">Grid: variant · score meter (0–1 scale) · outcome. The orange line marks the prune threshold (${p.threshold.toFixed(3)}); bars ending left of it are cut.</p>
  </div>`;
}

/* ── render: GoT aggregation diagram ─────────────────────── */
function renderAggregation(gg) {
  const a = gg.aggregation;
  if (!a.performed) {
    return `<div class="got-panel card">
      <h3>Aggregate</h3>
      <div class="agg-skip">Aggregation skipped — ${esc(a.skipped_reason || 'not enough distinct high-scoring designs to merge')}.</div>
    </div>`;
  }
  const sources = a.sources.map((s) =>
    `<div class="agg-node src"><span class="nm" title="${esc(prettyVariant(s.variant_type))}">${esc(prettyVariant(s.variant_type))}</span><span class="sc">${s.score.toFixed(3)}</span></div>`
  ).join('');

  const arrow = `<svg viewBox="0 0 64 90" fill="none" aria-hidden="true">
    ${[8, 28, 45, 62, 82].map((y) =>
      `<path d="M2 ${y} C 34 ${y}, 34 45, 56 45" stroke="var(--source)" stroke-width="1.5" opacity="0.6"/>`).join('')}
    <path d="M50 39 L58 45 L50 51" stroke="var(--source)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;

  return `<div class="got-panel card">
    <h3>Aggregate — ${a.sources.length} high-scorers merged into a hybrid</h3>
    <p class="p-sub">Candidates at or above 0.75 × best (${a.threshold.toFixed(3)}) are merged: compatible functions, behaviors and structures combine into one design, which is then re-scored.</p>
    <div class="agg">
      <div class="agg-sources">${sources}</div>
      <div class="agg-arrow">${arrow}</div>
      <div class="agg-result">
        <div class="hr">◇ Hybrid</div>
        <div class="num">${a.result.score.toFixed(3)}</div>
        <div class="cap">re-scored composite</div>
      </div>
    </div>
  </div>`;
}

function renderGoT(data) {
  const gg = data.got_graph;
  const section = $('got-section');
  if (!gg || !gg.enabled) { section.hidden = true; return; }
  $('got').innerHTML = renderPrune(gg) + renderAggregation(gg);
  section.hidden = false;
  positionThreshold();
}

/* place the threshold line precisely over the bar column after layout */
function positionThreshold() {
  const overlay = document.querySelector('.pbar-thr-overlay');
  const firstBar = document.querySelector('.prow .pbar-wrap');
  const rowsBox = document.querySelector('.prune-rows');
  const p = window.__lastGoT && window.__lastGoT.prune;
  if (!overlay || !firstBar || !rowsBox || !p) return;
  const bar = firstBar.getBoundingClientRect();
  const box = rowsBox.getBoundingClientRect();
  const left = (bar.left - box.left) + bar.width * Math.max(0, Math.min(1, p.threshold));
  overlay.className = 'pbar-thr';
  overlay.dataset.label = p.threshold.toFixed(2);
  overlay.style.cssText =
    `position:absolute;top:-6px;bottom:0;left:${left.toFixed(1)}px;width:2px;background:var(--source);z-index:4;`;
}

/* ── saved runs (structured store) ───────────────────────── */
let CURRENT_RUN_ID = null;

function fmtDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined,
    { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

async function refreshSaved() {
  const section = $('saved-section'), list = $('saved-list');
  try {
    const resp = await fetch('/results', { cache: 'no-store' });
    if (!resp.ok) throw new Error(resp.status);
    const data = await resp.json();
    const runs = data.runs || [];
    const st = data.stats || {};
    $('saved-stats').textContent = runs.length
      ? `${st.runs} run${st.runs === 1 ? '' : 's'} · ${st.prototypes} prototypes · ${(st.db_bytes / 1e6).toFixed(1)} MB`
      : 'No runs stored yet — generate one and it will be saved here.';

    list.innerHTML = runs.map((r) => `
      <div class="saved-row${r.run_id === CURRENT_RUN_ID ? ' active' : ''}" data-run="${esc(r.run_id)}">
        <div class="sr-main">
          <div class="sr-name">${esc(r.project_name || 'Untitled project')}</div>
          <div class="sr-brief" title="${esc(r.brief || '')}">${esc((r.brief || '(no brief recorded)').slice(0, 110))}${(r.brief || '').length > 110 ? '…' : ''}</div>
          <div class="sr-meta">${fmtDate(r.created_at)} · ${r.stored_prototypes} prototypes${
            r.top_score != null ? ` · top ${Number(r.top_score).toFixed(3)}` : ''}${
            r.complexity_level ? ` · ${esc(r.complexity_level)}` : ''}</div>
        </div>
        <div class="sr-actions">
          <button type="button" class="btn btn-ghost btn-sm" data-act="load">Open</button>
          <button type="button" class="icon-btn danger" data-act="del" title="Delete this run">✕</button>
        </div>
      </div>`).join('');
    section.hidden = false;

    list.querySelectorAll('.saved-row').forEach((row) => {
      const id = row.dataset.run;
      row.querySelector('[data-act="load"]').addEventListener('click', () => openRun(id));
      row.querySelector('[data-act="del"]').addEventListener('click', () => deleteRun(id, row));
    });
  } catch (err) {
    // No backend (static/sample mode) — keep the panel out of the way.
    section.hidden = true;
  }
}

async function openRun(runId) {
  setStatus('Loading stored run…', 'busy');
  try {
    const resp = await fetch(`/results/${encodeURIComponent(runId)}`, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    CURRENT_RUN_ID = runId;
    if (data.project_name) $('project_name').value = data.project_name;
    if (data.brief) $('requirements').value = data.brief;
    setStatus(`Loaded stored run — ${(data.designs || []).length} prototypes.`);
    render(data);
    refreshSaved();
  } catch (err) {
    setStatus('Could not load run: ' + err.message, 'error');
  }
}

async function deleteRun(runId, row) {
  const name = row.querySelector('.sr-name').textContent;
  if (!confirm(`Delete "${name}" and all of its prototypes?\n\nThis cannot be undone.`)) return;
  try {
    const resp = await fetch(`/results/${encodeURIComponent(runId)}`, { method: 'DELETE' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    if (runId === CURRENT_RUN_ID) {
      CURRENT_RUN_ID = null;
      ['summary', 'designs-section', 'got-section'].forEach((id) => { $(id).hidden = true; });
    }
    setStatus(`Deleted "${name}".`);
    refreshSaved();
  } catch (err) {
    setStatus('Delete failed: ' + err.message, 'error');
  }
}

async function deletePrototype(prototypeId, card) {
  if (!CURRENT_RUN_ID) return;
  if (!confirm('Delete this prototype from the stored run?\n\nThis cannot be undone.')) return;
  try {
    const resp = await fetch(
      `/results/${encodeURIComponent(CURRENT_RUN_ID)}/prototypes/${encodeURIComponent(prototypeId)}`,
      { method: 'DELETE' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    card.remove();
    setStatus('Prototype deleted.');
    refreshSaved();
  } catch (err) {
    setStatus('Delete failed: ' + err.message, 'error');
  }
}

/* ── main render ─────────────────────────────────────────── */
function render(data) {
  window.__lastGoT = data.got_graph;
  const topK = Number($('top_k').value) || 3;
  renderSummary(data);
  renderDesigns(data, topK);
  renderGoT(data);
  window.scrollTo({ top: statusEl.offsetTop - 12, behavior: 'smooth' });
}
window.addEventListener('resize', positionThreshold);

/* ── live run ────────────────────────────────────────────── */
form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const req = $('requirements').value.trim();
  if (!req) { setStatus('Enter a design brief first.', 'error'); return; }

  const payload = {
    project_name: $('project_name').value || undefined,
    requirements: req,
    use_got: $('use_got').checked,
  };
  // Only cap the candidate pool if a number is given; blank = complexity-adaptive.
  const maxc = $('max_alternatives').value.trim();
  if (maxc) payload.max_alternatives = Number(maxc);

  const btn = $('run-btn');
  btn.disabled = true;
  setStatus('Running the full pipeline — encoding → research → GoT exploration → scoring → layout. This can take 30 s–2 min…', 'busy');

  try {
    const resp = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      const t = await resp.text();
      setStatus(`Server error ${resp.status}: ${t.slice(0, 300)}`, 'error');
      return;
    }
    const data = await resp.json();
    if (!data.success) {
      setStatus('Pipeline error: ' + (data.error || 'unknown'), 'error');
      return;
    }
    setStatus(`Done — ${(data.designs || []).length} prototypes in ${Number(data.processing_time || 0).toFixed(1)}s.`);
    CURRENT_RUN_ID = data.run_id || null;
    render(data);
    refreshSaved();   // the run has just been persisted; show it in the list
  } catch (err) {
    setStatus('Could not reach the API (' + err.message + '). Start the backend with `uvicorn backend.main:app`, or use “Load sample run”.', 'error');
  } finally {
    btn.disabled = false;
  }
});

/* ── sample run ──────────────────────────────────────────── */
async function loadSample() {
  setStatus('Loading captured sample run…', 'busy');
  try {
    const resp = await fetch('sample_result.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error('sample_result.json not found');
    const data = await resp.json();
    if (data.project_name) $('project_name').value = data.project_name;
    setStatus(`Sample loaded — ${(data.designs || []).length} prototypes from a real captured run.`);
    render(data);
  } catch (err) {
    setStatus('Could not load sample_result.json (' + err.message + '). Serve this folder over HTTP rather than opening the file directly.', 'error');
  }
}
$('sample-btn').addEventListener('click', loadSample);

// Shareable demo link: /index.html?demo=1 auto-loads the captured sample run.
// Optional &theme=light|dark forces a mode (handy for previews/screenshots).
{
  const q = new URLSearchParams(location.search);
  if (q.get('theme') === 'light' || q.get('theme') === 'dark') {
    document.documentElement.setAttribute('data-theme', q.get('theme'));
  }
  if (q.has('demo')) {
    window.addEventListener('DOMContentLoaded', loadSample);
    loadSample();
  }
}

/* ── backend build ───────────────────────────────────────── */
async function showBuild() {
  // A running uvicorn holds the Python it imported at startup, while the
  // frontend is served from disk and updates on refresh. A new UI driving an
  // old backend looks exactly like a bug in the new code, so show the build.
  try {
    const r = await fetch('/health', { cache: 'no-store' });
    if (!r.ok) return;
    const h = await r.json();
    const el = document.getElementById('build-id');
    if (el && h.build) { el.textContent = `backend build ${h.build}`; el.hidden = false; }
  } catch (_) { /* static mode: no backend to report */ }
}
showBuild();

/* ── saved-runs panel: populate on load ──────────────────── */
$('refresh-saved').addEventListener('click', refreshSaved);
refreshSaved();
