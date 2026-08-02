"""Structured store for pipeline results.

Every run is persisted as a row in `runs` plus one row per prototype in
`prototypes`, each carrying the *complete* design: the full FBSL (functions,
behaviors, structures, layout), the layout geometry on its own, the connectivity
graph, all five sub-scores, and the rendered floor-plan/adjacency images.

**Why SQLite rather than the filesystem bundles this replaces.** Runs used to
land in `outputs/<uuid>/prototypes/...` as loose JSON and images, with the
throwaway renders piling up separately in `visual_outputs/` — 2,700 files and
431 MB of them by the time this was written, with no way to list what a run
contained or remove one you did not want. A single file-backed database gives
ordered listing, one-statement deletion with `ON DELETE CASCADE` cleaning up a
run's prototypes, and no server to run — `sqlite3` is in the standard library.

The store is deliberately self-contained: images are held as `data:` URIs
alongside the design they belong to, so deleting a run cannot leave orphaned
files behind.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "results" / "kags_results.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id             TEXT PRIMARY KEY,
    project_name       TEXT,
    brief              TEXT,
    created_at         TEXT NOT NULL,
    method             TEXT,
    complexity_level   TEXT,
    processing_time    REAL,
    n_designs          INTEGER,
    got_graph          TEXT,
    graph_statistics   TEXT,
    complexity_metrics TEXT
);

CREATE TABLE IF NOT EXISTS prototypes (
    prototype_id           TEXT PRIMARY KEY,
    run_id                 TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    rank                   INTEGER,
    node_id                TEXT,
    variant_type           TEXT,
    description            TEXT,
    composite              REAL,
    functional             REAL,
    behavioral             REAL,
    structural             REAL,
    layout_efficiency      REAL,
    sustainability         REAL,
    functions_count        INTEGER,
    behaviors_count        INTEGER,
    structures_count       INTEGER,
    converged              INTEGER,
    convergence_iterations INTEGER,
    fbsl                   TEXT,
    layout                 TEXT,
    connectivity           TEXT,
    png_floor_plan         TEXT,
    png_adjacency          TEXT,
    svg_floor_plan         TEXT,
    adjacency_svg          TEXT,
    created_at             TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_prototypes_run ON prototypes(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_created  ON runs(created_at DESC);
"""


class ResultStore:
    """File-backed store for runs and their prototypes."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.info(f"[OK] Result store ready at {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Required for ON DELETE CASCADE — SQLite leaves it off by default, and
        # without it deleting a run would silently orphan all its prototypes.
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------ write
    def save_run(self, result: Dict[str, Any], brief: str = "") -> Optional[str]:
        """Persist one pipeline result. Returns the run_id, or None if the
        result was unsuccessful (a failed run has nothing worth keeping)."""
        if not result or not result.get("success"):
            return None

        run_id = str(result.get("project_id") or uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        designs = result.get("designs") or []
        complexity = result.get("complexity_metrics") or {}

        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, project_name, brief, created_at, method, complexity_level,
                    processing_time, n_designs, got_graph, graph_statistics, complexity_metrics)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    result.get("project_name") or "Untitled project",
                    brief,
                    now,
                    result.get("method"),
                    (complexity or {}).get("level"),
                    result.get("processing_time"),
                    len(designs),
                    _dumps(result.get("got_graph")),
                    _dumps(result.get("graph_statistics")),
                    _dumps(complexity),
                ),
            )
            # Replacing a run replaces its prototypes wholesale.
            conn.execute("DELETE FROM prototypes WHERE run_id = ?", (run_id,))

            for rank, d in enumerate(designs, 1):
                scores = d.get("scores") or {}
                fbsl = d.get("fbsl") or {}
                layout = fbsl.get("layout") or {}
                conn.execute(
                    """INSERT INTO prototypes
                       (prototype_id, run_id, rank, node_id, variant_type, description,
                        composite, functional, behavioral, structural, layout_efficiency,
                        sustainability, functions_count, behaviors_count, structures_count,
                        converged, convergence_iterations, fbsl, layout, connectivity,
                        png_floor_plan, png_adjacency, svg_floor_plan, adjacency_svg, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(uuid.uuid4()), run_id, rank,
                        d.get("node_id"), d.get("variant_type"), d.get("description"),
                        scores.get("composite"), scores.get("functional_adequacy"),
                        scores.get("behavioral_performance"), scores.get("structural_feasibility"),
                        scores.get("layout_efficiency"), scores.get("sustainability"),
                        d.get("functions_count"), d.get("behaviors_count"), d.get("structures_count"),
                        1 if d.get("converged") else 0, d.get("convergence_iterations"),
                        _dumps(fbsl), _dumps(layout), _dumps(_connectivity_of(layout)),
                        d.get("png_floor_plan"), d.get("png_adjacency"),
                        d.get("svg_floor_plan"), d.get("adjacency_svg"),
                        now,
                    ),
                )
        logger.info(f"[OK] Stored run {run_id[:8]} with {len(designs)} prototypes")
        return run_id

    # ------------------------------------------------------------------- read
    def list_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Run summaries, newest first. Deliberately excludes the heavy columns
        so the UI's list view never transfers image payloads."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT r.run_id, r.project_name, r.brief, r.created_at, r.method,
                          r.complexity_level, r.processing_time, r.n_designs,
                          (SELECT COUNT(*) FROM prototypes p WHERE p.run_id = r.run_id)
                              AS stored_prototypes,
                          (SELECT MAX(composite) FROM prototypes p WHERE p.run_id = r.run_id)
                              AS top_score
                   FROM runs r ORDER BY r.created_at DESC, r.rowid DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """A full run, shaped like a live pipeline result so the same UI code
        renders either without a translation layer."""
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            protos = conn.execute(
                "SELECT * FROM prototypes WHERE run_id = ? ORDER BY rank", (run_id,)
            ).fetchall()

        return {
            "success": True,
            "stored": True,
            "run_id": run["run_id"],
            "project_id": run["run_id"],
            "project_name": run["project_name"],
            "brief": run["brief"],
            "created_at": run["created_at"],
            "method": run["method"],
            "processing_time": run["processing_time"],
            "got_graph": _loads(run["got_graph"]),
            "graph_statistics": _loads(run["graph_statistics"]),
            "complexity_metrics": _loads(run["complexity_metrics"]),
            "designs": [_row_to_design(p) for p in protos],
        }

    # ----------------------------------------------------------------- delete
    def delete_run(self, run_id: str) -> bool:
        """Delete a run and, by cascade, every prototype belonging to it."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        deleted = cur.rowcount > 0
        if deleted:
            logger.info(f"Deleted run {run_id[:8]} and its prototypes")
        return deleted

    def delete_prototype(self, prototype_id: str) -> bool:
        """Delete a single prototype, leaving the rest of its run intact."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM prototypes WHERE prototype_id = ?", (prototype_id,))
        return cur.rowcount > 0

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            runs = conn.execute("SELECT COUNT(*) c FROM runs").fetchone()["c"]
            protos = conn.execute("SELECT COUNT(*) c FROM prototypes").fetchone()["c"]
        size = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {"runs": runs, "prototypes": protos, "db_bytes": size,
                "db_path": str(self.db_path)}


# ---------------------------------------------------------------- helpers
def _dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _loads(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _connectivity_of(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the connectivity graph out of a layout as nodes + edges.

    The layout carries adjacency as per-room lists; storing an explicit
    node/edge form means a consumer does not have to re-derive the graph, and
    the required-vs-achieved distinction survives (an edge the brief asked for
    but the plan did not deliver is exactly what a reviewer wants to see).
    """
    rooms = (layout or {}).get("rooms") or {}
    nodes, edges, seen = [], [], set()
    for rid, room in rooms.items():
        nodes.append({
            "id": rid,
            "name": room.get("name"),
            "room_type": room.get("room_type"),
            "area": room.get("area"),
        })
        for kind in ("required_adjacencies", "actual_adjacencies"):
            for other in (room.get(kind) or []):
                key = tuple(sorted((str(rid), str(other)))) + (kind,)
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"source": rid, "target": other,
                              "kind": "required" if kind.startswith("required") else "actual"})
    return {"nodes": nodes, "edges": edges}


def _row_to_design(p: sqlite3.Row) -> Dict[str, Any]:
    """Shape a stored prototype like a live `designs[]` entry."""
    return {
        "prototype_id": p["prototype_id"],
        "node_id": p["node_id"],
        "rank": p["rank"],
        "variant_type": p["variant_type"],
        "description": p["description"],
        "scores": {
            "composite": p["composite"],
            "functional_adequacy": p["functional"],
            "behavioral_performance": p["behavioral"],
            "structural_feasibility": p["structural"],
            "layout_efficiency": p["layout_efficiency"],
            "sustainability": p["sustainability"],
        },
        "functions_count": p["functions_count"],
        "behaviors_count": p["behaviors_count"],
        "structures_count": p["structures_count"],
        "converged": bool(p["converged"]),
        "convergence_iterations": p["convergence_iterations"],
        "fbsl": _loads(p["fbsl"]),
        "layout": _loads(p["layout"]),
        "connectivity": _loads(p["connectivity"]),
        "png_floor_plan": p["png_floor_plan"],
        "png_adjacency": p["png_adjacency"],
        "svg_floor_plan": p["svg_floor_plan"],
        "adjacency_svg": p["adjacency_svg"],
    }
