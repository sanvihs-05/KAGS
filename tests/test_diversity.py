"""Design-space diversity: signature fingerprinting, clone dedup, and the five
named Level-1 GoT strategies each carrying real parameter deltas."""
import asyncio
import copy
from backend.core.fbsl_models import (
    FBSLLayoutNode, Function, Room, Layout, Structure,
    StructureType, FunctionCategory,
)
from backend.core.design_signature import design_signature, dedupe_by_signature
from backend.core.graph_of_thoughts import GraphOfThoughtsEngine

PROGRAM = [("Living", "living_room", 40), ("Kitchen", "kitchen", 16),
           ("Master Bedroom", "bedroom", 20), ("Bedroom 2", "bedroom", 14),
           ("Bathroom", "bathroom", 6)]


def _node():
    n = FBSLLayoutNode(); n.layout = Layout()
    for name, rt, a in PROGRAM:
        f = Function(name=f"provide_{rt}", category=FunctionCategory.SPATIAL, priority=0.8,
                     spatial_requirements={'min_area': a * 0.7, 'max_area': a * 1.3})
        n.functions[f.function_id] = f
        r = Room(name=name, room_type=rt, area=a, function_id=f.function_id)
        n.layout.rooms[r.room_id] = r
    n.layout.total_area = sum(p[2] for p in PROGRAM)
    for sn, st, mat, cat, dims, lb in [
        ("window", StructureType.WALL, "glazing", "envelope", {'window_ratio': 0.15}, False),
        ("hvac", StructureType.MEP, "steel", "services", {'flow_rate': 0.5}, False),
        ("foundation", StructureType.FOUNDATION, "concrete", "structural", {'thickness': 0.3}, True),
    ]:
        s = Structure(name=sn, structure_type=st, material_type=mat, category=cat,
                      dimensions=dict(dims), load_bearing=lb)
        n.structures[s.structure_id] = s
    return n


def test_signature_matches_clones_separates_variants():
    a = _node(); b = copy.deepcopy(a)
    assert design_signature(a) == design_signature(b), "identical nodes share a signature"
    b.metadata['layout_aspect'] = 2.4
    assert design_signature(a) != design_signature(b), "different aspect => different signature"


def test_dedupe_collapses_clones():
    a = _node()
    pool = [a, copy.deepcopy(a), copy.deepcopy(a)]
    assert len(dedupe_by_signature(pool)) == 1


def test_five_named_strategies_are_distinct():
    engine = GraphOfThoughtsEngine(max_depth=2, breadth=4, encoder=None)
    seeds = asyncio.run(engine._strategy_seeds(_node()))
    names = {s.metadata.get('variant_type') for s in seeds}
    assert {'functional_priority', 'performance_optimized', 'structural_efficiency',
            'spatial_compactness', 'balanced'} <= names
    # every strategy is a genuinely distinct design
    assert len({design_signature(s) for s in seeds}) == len(seeds)


def test_natural_ventilation_variant_drops_mep():
    engine = GraphOfThoughtsEngine(max_depth=2, breadth=4, encoder=None)
    variants = asyncio.run(engine._behavioral_optimization(_node()))
    nat = [v for v in variants if v.metadata.get('ventilation_strategy') == 'natural']
    assert len(nat) == 1
    assert not any(getattr(s.structure_type, 'value', '') == 'mep'
                   for s in nat[0].structures.values())
