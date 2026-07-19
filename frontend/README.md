KAGS frontend — simple UI

This is a minimal frontend to call the backend pipeline API and display the top prototypes.

How to use

1. Start the backend API (assumes `uvicorn` and that `backend/main.py` registers `/pipeline/run`):

```powershell
# from repo root
py -m uvicorn backend.main:app --reload --port 8000
```

2. Serve the frontend folder or open `index.html` in a browser.

Quick serve (recommended) — run a simple static server from the repository root:

```powershell
# from repo root
py -m http.server 3000
# then open http://localhost:3000/frontend/
```

Notes
- If your backend is on a different host/port, edit `frontend/app.js` and change `API_URL` to the full URL (e.g. `http://localhost:8000/pipeline/run`).
- If CORS is not enabled on the backend you'll need to enable it or run the frontend served from the same origin.
- The app inserts returned SVG strings directly into the page. Only use with trusted local backend responses.

