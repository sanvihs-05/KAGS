const API_URL = (window.API_URL || '/pipeline/run');

const form = document.getElementById('pipeline-form');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');

function setStatus(msg, isError=false) {
  statusEl.textContent = msg;
  statusEl.className = isError ? 'status error' : 'status';
}

function clearResults() {
  resultsEl.innerHTML = '';
}

function makeCard(design) {
  const container = document.createElement('article');
  container.className = 'card';

  const header = document.createElement('header');
  header.innerHTML = `<h3>Node ${design.node_id}</h3>`;
  container.appendChild(header);

  const scores = design.scores || {};
  const scoreList = document.createElement('ul');
  scoreList.className = 'scores';
  for (const [k,v] of Object.entries(scores)) {
    const li = document.createElement('li');
    li.textContent = `${k}: ${Number(v).toFixed(3)}`;
    scoreList.appendChild(li);
  }
  container.appendChild(scoreList);

  if (design.svg_floor_plan) {
    const svgWrap = document.createElement('div');
    svgWrap.className = 'svg-wrap';
    // Insert SVG markup directly
    svgWrap.innerHTML = '<h4>Floor Plan</h4>' + design.svg_floor_plan;
    container.appendChild(svgWrap);
  }

  if (design.adjacency_svg) {
    const adjWrap = document.createElement('div');
    adjWrap.className = 'svg-wrap';
    adjWrap.innerHTML = '<h4>Adjacency Graph</h4>' + design.adjacency_svg;
    container.appendChild(adjWrap);
  }

  // Show metadata JSON (collapsible)
  const metaPre = document.createElement('pre');
  metaPre.textContent = JSON.stringify(design.metadata || {}, null, 2);
  container.appendChild(metaPre);

  return container;
}

form.addEventListener('submit', async (ev) => {
  ev.preventDefault();
  clearResults();
  setStatus('Submitting request...', false);

  const payload = {
    project_name: document.getElementById('project_name').value || undefined,
    requirements: document.getElementById('requirements').value,
    max_alternatives: Number(document.getElementById('max_alternatives').value) || 5,
    use_got: true
  };

  // Optional params
  const delta = document.getElementById('got_delta').value;
  if (delta) payload.got_delta = Number(delta);
  const patience = document.getElementById('got_patience').value;
  if (patience) payload.got_patience = Number(patience);
  const max_nodes = document.getElementById('got_max_nodes').value;
  if (max_nodes) payload.got_max_nodes = Number(max_nodes);
  const sel = document.getElementById('got_selection_metric').value;
  if (sel) payload.got_selection_metric = sel;

  try {
    setStatus('Running pipeline (this may take 10s–minutes)...');
    const resp = await fetch(API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!resp.ok) {
      const text = await resp.text();
      setStatus(`Server error: ${resp.status} ${resp.statusText} - ${text}`, true);
      return;
    }

    const data = await resp.json();
    if (!data.success) {
      setStatus('Pipeline returned an error: ' + (data.error || JSON.stringify(data)), true);
      return;
    }

    const designs = data.designs || [];
    setStatus(`Pipeline finished — ${designs.length} designs returned`);

    const topK = Number(document.getElementById('top_k').value) || 3;
    designs.slice(0, topK).forEach(d => {
      const card = makeCard(d);
      resultsEl.appendChild(card);
    });

  } catch (err) {
    setStatus('Client error: ' + err.message, true);
    console.error(err);
  }
});
