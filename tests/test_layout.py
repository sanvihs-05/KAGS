"""Layout agent: gap-free treemap placement, measured brief-adjacency
satisfaction, room-graph circulation (nonzero + footprint-sensitive), and the
FBSL L layer (coordinates + adjacency) persisted onto the node."""
import asyncio
from backend.core.fbsl_models import (
    FBSLLayoutNode, Function, Room, Layout, FunctionCategory,
)
from backend.agents.layout_agent import LayoutGenerationAgent

agent = LayoutGenerationAgent()

PROGRAM = [("Living/Dining", "living_room", 40), ("Kitchen", "kitchen", 16),
           ("Dining", "dining", 12), ("Master Bedroom", "bedroom", 20),
           ("Bedroom 2", "bedroom", 14), ("Bathroom", "bathroom", 6),
           ("Mudroom", "mudroom", 6), ("Garage", "garage", 24)]
REQS = [{'room1': 'kitchen', 'room2': 'dining', 'type': 'required'},
        {'room1': 'mudroom', 'room2': 'garage', 'type': 'required'},
        {'room1': 'bedroom', 'room2': 'bathroom', 'type': 'required'}]


def _node(aspect=1.2, reqs=None):
    n = FBSLLayoutNode(); n.layout = Layout()
    for name, rt, a in PROGRAM:
        f = Function(name=f"provide_{rt}", category=FunctionCategory.SPATIAL, priority=0.8,
                     spatial_requirements={'min_area': a * 0.7, 'max_area': a * 1.3})
        n.functions[f.function_id] = f
        r = Room(name=name, room_type=rt, area=a, function_id=f.function_id)
        n.layout.rooms[r.room_id] = r
    n.layout.total_area = sum(p[2] for p in PROGRAM)
    n.metadata['required_adjacencies'] = reqs if reqs is not None else []
    n.metadata['layout_aspect'] = aspect
    return n


def test_brief_adjacency_satisfied_and_scored():
    n = _node(reqs=REQS)
    layout = asyncio.run(agent.generate_layout(n))
    assert layout.metadata.get('adjacency_measured') is True
    assert layout.adjacency_satisfaction_score >= 2 / 3   # most/all requirements met
    details = layout.metadata.get('adjacency_requirements', [])
    assert details and all('satisfied' in d for d in details)


def test_rooms_carry_L_coordinates_and_adjacency():
    n = _node(reqs=REQS)
    asyncio.run(agent.generate_layout(n))
    rooms = n.layout.rooms.values()
    assert all(r.position_vector and 'x' in r.position_vector for r in rooms), \
        "every room must carry coordinates"
    assert sum(1 for r in rooms if r.actual_adjacencies) >= len(n.layout.rooms) - 1


def test_circulation_nonzero_and_footprint_sensitive():
    compact = _node(aspect=1.05)
    linear = _node(aspect=2.4)
    lc = asyncio.run(agent.generate_layout(compact))
    ll = asyncio.run(agent.generate_layout(linear))
    assert lc.circulation_efficiency > 0.0 and ll.circulation_efficiency > 0.0
    assert lc.metadata.get('circulation_measured') is True
    # compact vs linear must differ in circulation and/or compactness
    assert (abs(lc.circulation_efficiency - ll.circulation_efficiency) > 1e-3
            or abs(lc.compactness_score - ll.compactness_score) > 1e-3)


def test_to_dict_contains_full_L():
    n = _node(reqs=REQS)
    layout = asyncio.run(agent.generate_layout(n))
    n.layout = layout
    d = n.to_dict()['layout']
    assert d['actual_adjacency_matrix'] is not None
    assert d['room_order']
    any_room = next(iter(d['rooms'].values()))
    assert 'position_vector' in any_room
