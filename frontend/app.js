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
}

/* ── render: GoT prune chart ─────────────────────────────── */
function renderPrune(gg) {
  const p = gg.prune;
  const sourceIds = new Set((gg.aggregation.sources || []).map((s) => s.id));

  // Two real cut mechanisms: (1) score below the 0.70×best threshold,
  // (2) above threshold but dropped by the diversity / top-N cap.
  const rows = gg.candidates.map((c) => {
    const belowThr = c.score < p.threshold;
    const reason = c.kept ? 'kept' : (belowThr ? 'threshold' : 'cap');
    const cls = [c.kept ? 'kept' : 'pruned'];
    if (c.kept && sourceIds.has(c.id)) cls.push('source');
    const w = Math.max(0, Math.min(100, c.score * 100));
    const tag = { kept: ['k', 'KEPT'], threshold: ['p', '&lt; thr'], cap: ['c', 'cap'] }[reason];
    return `<div class="prow ${cls.join(' ')} r-${reason}">
      <span class="pv" title="${esc(prettyVariant(c.variant_type))}">${esc(prettyVariant(c.variant_type))}</span>
      <div class="pbar-wrap">
        <div class="pbar-track"></div>
        <div class="pbar-fill" style="width:${w}%"></div>
      </div>
      <span class="ptag"><span class="sc">${c.score.toFixed(3)}</span><span class="st ${tag[0]}">${tag[1]}</span></span>
    </div>`;
  }).join('');

  return `<div class="got-panel card">
    <h3>Prune — ${p.n_scored} candidates scored → ${p.n_kept} kept</h3>
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
    render(data);
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
