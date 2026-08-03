"""Precedent adjacency as a placement signal.

The CubiCasa corpus's genuinely relational knowledge — P(a-b adjacent | both
present) over 3,787 real plans — was loaded, thresholded and then only surfaced
as advisory text. Meanwhile RAG acted on the corpus's *least* informative
signal, mean room area, which the scorer cannot reward: precedent-grounded sizes
score no better than brief-derived ones, they just perturb the geometry.

Adjacency is different: `adjacency_satisfaction` is a real term in S_l. But the
requirement set it is measured against must stay the brief's, or the system ends
up grading itself against constraints it invented. So precedent orders rooms and
the brief remains the yardstick — these tests pin that split.
"""
from types import SimpleNamespace as NS

from backend.agents.layout_agent import LayoutGenerationAgent as LA


def _rooms(*types):
    return {f"r{i}": NS(room_type=t) for i, t in enumerate(types)}


def _items(rooms, areas):
    return [(rid, areas[i]) for i, rid in enumerate(rooms)]


def test_brief_requirement_pulls_its_partner_adjacent():
    rooms = _rooms("kitchen", "bedroom", "dining")
    items = _items(rooms, [16.0, 14.0, 12.0])
    ordered = LA._pair_aware_order(items, rooms,
                                   [("kitchen", "dining", "required")])
    types = [rooms[rid].room_type for rid, _ in ordered]
    assert abs(types.index("kitchen") - types.index("dining")) == 1


def test_precedent_pair_also_pulls_its_partner_adjacent():
    """With nothing stated by the brief, precedent should still shape the plan."""
    rooms = _rooms("kitchen", "bedroom", "dining")
    items = _items(rooms, [16.0, 14.0, 12.0])
    ordered = LA._pair_aware_order(items, rooms, [],
                                   [("kitchen", "dining", "preferred")])
    types = [rooms[rid].room_type for rid, _ in ordered]
    assert abs(types.index("kitchen") - types.index("dining")) == 1


def test_the_brief_wins_when_the_two_disagree():
    """Precedent is a tie-break, not a competitor. The brief's partner must be
    pulled in first, so precedent can never displace a stated requirement."""
    rooms = _rooms("kitchen", "bedroom", "dining")
    items = _items(rooms, [16.0, 14.0, 12.0])
    ordered = LA._pair_aware_order(
        items, rooms,
        [("kitchen", "bedroom", "required")],     # brief: kitchen-bedroom
        [("kitchen", "dining", "preferred")])     # precedent: kitchen-dining
    types = [rooms[rid].room_type for rid, _ in ordered]
    assert abs(types.index("kitchen") - types.index("bedroom")) == 1


def test_every_room_is_placed_exactly_once():
    """Ordering must be a permutation — a duplicated or dropped room would
    corrupt the tiling."""
    rooms = _rooms("kitchen", "bedroom", "dining", "bathroom", "living_room")
    items = _items(rooms, [16.0, 14.0, 12.0, 6.0, 30.0])
    ordered = LA._pair_aware_order(
        items, rooms,
        [("kitchen", "dining", "required")],
        [("living_room", "bathroom", "preferred")])
    assert sorted(rid for rid, _ in ordered) == sorted(rooms)
    assert len(ordered) == len(items)


def test_precedent_pairs_are_read_from_research_metadata():
    node = NS(metadata={"precedent_adjacencies": [
        {"room1": "Kitchen", "room2": "Dining", "probability": 0.82},
        {"room1": "x", "room2": "x"},        # degenerate, must be dropped
        "not-a-dict",                        # malformed, must be dropped
    ]})
    pairs = LA._precedent_preferred_pairs(node)
    assert pairs == [("kitchen", "dining", "preferred")]


def test_no_precedent_metadata_is_a_safe_no_op():
    assert LA._precedent_preferred_pairs(NS(metadata={})) == []
    assert LA._precedent_preferred_pairs(NS(metadata=None)) == []


def test_precedent_pairs_never_reach_the_scored_requirement_set():
    """The guard that keeps this honest: adjacency satisfaction is measured
    against the brief only, so precedent cannot inflate the denominator."""
    node = NS(metadata={
        "required_adjacencies": [{"room1": "kitchen", "room2": "dining"}],
        "precedent_adjacencies": [{"room1": "living_room", "room2": "balcony"}],
    })
    required = LA._brief_required_pairs(node)
    preferred = LA._precedent_preferred_pairs(node)
    assert required == [("kitchen", "dining", "required")]
    assert ("living_room", "balcony", "preferred") in preferred
    # the precedent pair must NOT appear among the scored requirements
    assert all(p[2] == "required" for p in required)
    assert not any(p[:2] == ("living_room", "balcony") for p in required)
