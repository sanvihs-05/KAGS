# FBSL-KAGS frontend

A single-page UI for the layout pipeline. You enter a design brief; it shows the
ranked **top outputs** (each with score meters and a generated floor-plan SVG) and
a visualization of the real **Graph-of-Thoughts** prune + aggregate steps.

## Files
- `index.html` / `styles.css` / `app.js` — the app (no build step, no dependencies).
- `sample_result.json` — a **real captured pipeline run** (3-bedroom townhouse), used
  by "Load sample run" so the UI works instantly and offline. Regenerate it by
  saving the dict returned by `orchestrator.process_design_request(...)` to this path.

## Run it

**Served by the backend (live pipeline).** From the repo root:

```powershell
py -m uvicorn backend.main:app --reload --port 8000
# then open  http://localhost:8000/frontend/
```

The form POSTs to `/pipeline/run`; a live run executes the full multi-agent
pipeline and takes ~30 s–2 min. Results (scores, floor-plan/adjacency SVGs, and the
`got_graph` prune/aggregate trace) come straight from the orchestrator.

**Static only (sample data, no backend).** Serve this folder over HTTP so
`fetch` can read `sample_result.json` (opening the file directly won't work):

```powershell
py -m http.server 3000    # from the repo root
# then open  http://localhost:3000/frontend/
```

Click **Load sample run**, or use the shareable demo link
`index.html?demo=1` (append `&theme=light` or `&theme=dark` to force a mode).

## Notes
- The GoT panel is driven entirely by `result.got_graph` (candidates + scores, the
  0.70×max prune threshold, and the aggregated hybrid). Every number is measured
  from the run, not illustrative.
- If the backend runs on a different origin, set `window.API_URL` before `app.js`
  loads, and make sure CORS is enabled (it is, in `backend/main.py`).
- The app injects backend-provided SVG strings directly — use only with your own
  trusted local backend.
