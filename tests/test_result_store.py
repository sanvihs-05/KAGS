"""Structured result store: round-trip, completeness, and deletion.

Runs used to land as loose files under `outputs/<uuid>/` with throwaway renders
piling up in `visual_outputs/` — 2,700 files and 431 MB of them, with no way to
list what a run contained or remove one. These tests pin the properties that
replacement has to hold: a stored prototype is *complete* (full FBSL, layout,
connectivity graph and images), and deleting a run takes its prototypes with it
rather than orphaning them.
"""
import uuid

import pytest

from backend.database.result_store import ResultStore, _connectivity_of


def _result(project_name="Test project", n=3):
    """A minimal result shaped like the orchestrator's return value."""
    rooms = {
        "r1": {"name": "Living", "room_type": "living_room", "area": 40.0,
               "required_adjacencies": ["r2"], "actual_adjacencies": ["r2"]},
        "r2": {"name": "Kitchen", "room_type": "kitchen", "area": 16.0,
               "required_adjacencies": [], "actual_adjacencies": ["r1"]},
    }
    return {
        "success": True,
        "project_id": str(uuid.uuid4()),
        "project_name": project_name,
        "method": "Graph of Thought",
        "processing_time": 12.5,
        "complexity_metrics": {"level": "medium", "room_count": 2},
        "got_graph": {"enabled": True, "prune": {"n_scored": 5, "n_kept": 3}},
        "graph_statistics": {"total_nodes": 9},
        "designs": [{
            "node_id": f"node{i}",
            "variant_type": f"variant_{i}",
            "description": "d",
            "scores": {"composite": 0.9 - i / 100, "functional_adequacy": 0.8,
                       "behavioral_performance": 0.85, "structural_feasibility": 0.94,
                       "layout_efficiency": 0.97, "sustainability": 0.5},
            "functions_count": 2, "behaviors_count": 4, "structures_count": 6,
            "converged": True, "convergence_iterations": 2,
            "fbsl": {"functions": {"f1": {}}, "behaviors": {"b1": {}},
                     "structures": {"s1": {}}, "layout": {"rooms": rooms}},
            "png_floor_plan": "data:image/png;base64,AAAA",
            "png_adjacency": "data:image/png;base64,BBBB",
            "svg_floor_plan": "<svg/>", "adjacency_svg": "<svg/>",
        } for i in range(n)],
    }


@pytest.fixture
def store(tmp_path):
    return ResultStore(tmp_path / "t.db")


def test_round_trip_preserves_the_whole_design(store):
    """A prototype must come back complete — the point of the store is that
    nothing about a design lives anywhere else."""
    run_id = store.save_run(_result(), brief="a brief")
    run = store.get_run(run_id)

    assert run["project_name"] == "Test project"
    assert run["brief"] == "a brief"
    assert run["got_graph"]["prune"]["n_kept"] == 3      # run-level trace survives
    p = run["designs"][0]
    assert set(p["fbsl"]) >= {"functions", "behaviors", "structures", "layout"}
    assert len(p["layout"]["rooms"]) == 2
    assert p["connectivity"]["nodes"] and p["connectivity"]["edges"]
    assert p["png_floor_plan"] and p["png_adjacency"]
    assert p["svg_floor_plan"] and p["adjacency_svg"]
    assert p["scores"]["composite"] == pytest.approx(0.9)


def test_connectivity_graph_distinguishes_required_from_actual():
    """A link the brief asked for but the plan did not deliver is exactly what a
    reviewer needs to see, so the two edge kinds must not be collapsed."""
    conn = _connectivity_of({"rooms": {
        "a": {"room_type": "kitchen", "required_adjacencies": ["b"], "actual_adjacencies": []},
        "b": {"room_type": "dining", "required_adjacencies": [], "actual_adjacencies": []},
    }})
    kinds = {e["kind"] for e in conn["edges"]}
    assert kinds == {"required"}
    assert {n["id"] for n in conn["nodes"]} == {"a", "b"}


def test_runs_are_listed_newest_first_without_image_payloads(store):
    store.save_run(_result("First"))
    store.save_run(_result("Second"))
    rows = store.list_runs()
    assert [r["project_name"] for r in rows][:2] == ["Second", "First"]
    # the list view must stay light — no blob columns
    assert "png_floor_plan" not in rows[0]
    assert rows[0]["stored_prototypes"] == 3


def test_deleting_a_run_cascades_to_its_prototypes(store):
    """Without PRAGMA foreign_keys=ON SQLite silently orphans the children, so
    this asserts the cascade actually fires."""
    run_id = store.save_run(_result())
    assert store.stats()["prototypes"] == 3
    assert store.delete_run(run_id) is True
    assert store.stats() == {**store.stats(), "runs": 0, "prototypes": 0}
    assert store.get_run(run_id) is None


def test_deleting_one_prototype_leaves_the_rest_of_the_run(store):
    run_id = store.save_run(_result(n=3))
    victim = store.get_run(run_id)["designs"][1]["prototype_id"]
    assert store.delete_prototype(victim) is True
    remaining = store.get_run(run_id)["designs"]
    assert len(remaining) == 2
    assert victim not in {p["prototype_id"] for p in remaining}


def test_deleting_something_absent_reports_false_rather_than_raising(store):
    assert store.delete_run("no-such-run") is False
    assert store.delete_prototype("no-such-prototype") is False


def test_failed_runs_are_not_stored(store):
    """A failed run has nothing worth keeping and must not clutter the list."""
    assert store.save_run({"success": False, "error": "boom"}) is None
    assert store.save_run(None) is None
    assert store.list_runs() == []


def test_resaving_a_run_replaces_its_prototypes_rather_than_duplicating(store):
    result = _result(n=3)
    store.save_run(result)
    store.save_run(result)          # same project_id
    assert store.stats()["runs"] == 1
    assert store.stats()["prototypes"] == 3
