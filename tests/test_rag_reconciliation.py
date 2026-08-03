"""RAG area reconciliation: precedent fills gaps, the brief wins where explicit.

Reconciliation used to blend *every* room toward the corpus mean at lambda 0.6,
including sizes the client had stated, and it moved the room without moving the
area Behavior's target — so the design ended up missing its own spec and S_f/S_b
dropped by construction. The ablation showed it: removing RAG *improved*
composite by 1.57 % and 1.80 % on the two briefs that state areas, and was noise
(-0.11 %) on the vague one that states none.
"""
from types import SimpleNamespace as NS

import pytest

from backend.agents.research_agent import ResearchAgent


class _Room:
    def __init__(self, area, from_default):
        self.area = area
        self.function_id = "f1"
        self.metadata = {"area_from_default": from_default}


def _node(area=24.0, from_default=False, target=24.0):
    func = NS(function_id="f1", name="provide_bedroom",
              spatial_requirements={"min_area": 8.0, "max_area": 40.0,
                                    "preferred_area": area})
    behaviour = NS(derived_from_function="f1", metric_name="bedroom_area",
                   target_value=target)
    room = _Room(area, from_default)
    return NS(functions={"f1": func},
              behaviors={"b1": behaviour},
              layout=NS(rooms={"r1": room}))


def _findings(precedent_area=12.0):
    return {"room_precedents": {"provide_bedroom": [
        {"area": precedent_area, "similarity": 0.9},
    ]}}


def test_a_brief_stated_area_is_never_overridden():
    """The client said 24 m². Precedent averaging must not quietly make it 19."""
    node = _node(area=24.0, from_default=False)
    n = ResearchAgent.reconcile_areas_with_precedents(None, node, _findings(12.0))
    assert n == 0
    assert node.layout.rooms["r1"].area == 24.0


def test_a_defaulted_area_is_grounded_in_precedent():
    """Where the brief said nothing, precedent is exactly what should fill in."""
    node = _node(area=24.0, from_default=True)
    n = ResearchAgent.reconcile_areas_with_precedents(None, node, _findings(12.0))
    assert n == 1
    # lambda 0.6: 0.6*24 + 0.4*12 = 19.2
    assert node.layout.rooms["r1"].area == pytest.approx(19.2, abs=0.01)


def test_the_behaviour_target_moves_with_the_room():
    """Moving the design without moving the yardstick made the design miss its
    own spec — the mechanism behind RAG scoring as a net negative."""
    node = _node(area=24.0, from_default=True, target=24.0)
    ResearchAgent.reconcile_areas_with_precedents(None, node, _findings(12.0))
    room_area = node.layout.rooms["r1"].area
    assert node.behaviors["b1"].target_value == pytest.approx(room_area, abs=0.01)


def test_reconciliation_stays_inside_the_brief_band():
    """Clamping to [min_area, max_area] keeps the brief validator satisfiable."""
    node = _node(area=10.0, from_default=True)
    node.functions["f1"].spatial_requirements.update({"min_area": 9.0, "max_area": 11.0})
    ResearchAgent.reconcile_areas_with_precedents(None, node, _findings(200.0))
    assert 9.0 <= node.layout.rooms["r1"].area <= 11.0


def test_no_precedents_is_a_safe_no_op():
    node = _node(from_default=True)
    assert ResearchAgent.reconcile_areas_with_precedents(None, node, {}) == 0
    assert ResearchAgent.reconcile_areas_with_precedents(
        None, node, {"room_precedents": {}}) == 0


# ---------------------------------------------------------------- ranking
def _func(preferred):
    return NS(spatial_requirements={"preferred_area": preferred})


def test_area_proximity_reranks_same_type_precedents():
    """Retrieval is semantic now, so every precedent returns a similar score and
    the ordering carries no size information until this runs."""
    precedents = [
        {"area": 40.0, "similarity": 0.90},
        {"area": 16.0, "similarity": 0.88},   # closest to target, worse raw score
        {"area": 30.0, "similarity": 0.89},
    ]
    ranked = ResearchAgent._rank_by_area_proximity(precedents, _func(16.0))
    assert ranked[0]["area"] == 16.0


def test_area_proximity_is_monotonic_in_area_error():
    """The property the embedding failed to provide: closer area, higher score."""
    scored = ResearchAgent._rank_by_area_proximity(
        [{"area": a, "similarity": 1.0} for a in (16.0, 20.0, 25.0, 40.0)], _func(16.0))
    areas = [p["area"] for p in scored]
    assert areas == [16.0, 20.0, 25.0, 40.0]


def test_precedents_without_an_area_are_left_alone():
    """Stores that expose no area must degrade to the previous behaviour."""
    precedents = [{"similarity": 0.8}, {"similarity": 0.9}]
    ranked = ResearchAgent._rank_by_area_proximity(precedents, _func(16.0))
    assert all("area_score" not in p for p in ranked)


def test_ranking_without_a_target_is_a_no_op():
    original = [{"area": 40.0, "similarity": 0.5}]
    assert ResearchAgent._rank_by_area_proximity(original, NS(spatial_requirements={})) is original
