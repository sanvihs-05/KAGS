"""Scoring: S_sust and S_s must be REAL (computed from the design), not the
flat constants they used to be. Guards against regressing back to 0.5 / 1.0."""
from backend.core.fbsl_models import (
    FBSLLayoutNode, Structure, Layout, Room, StructureType, Behavior, BehaviorCategory,
)
from backend.agents.scoring_agent import ScoringAgent

scorer = ScoringAgent(rho=1.0)


def _behav_node(ratio):
    """A node whose behaviors all perform at `ratio` × their target."""
    n = FBSLLayoutNode()
    for cat, tgt in [(BehaviorCategory.THERMAL, 21), (BehaviorCategory.LIGHTING, 3),
                     (BehaviorCategory.ACOUSTIC, 45)]:
        b = Behavior(category=cat, metric_name=f"{cat.value}_m", metric_unit="x",
                     target_value=tgt, actual_value=tgt * ratio, tolerance=0.2)
        n.behaviors[b.behavior_id] = b
    return n


def test_behavioral_rewards_exceeding_target():
    """S_b must distinguish deficient < adequate < excellent — the old
    min(1, actual/target) flattened adequate and excellent both to 1.0."""
    deficient, _ = scorer._score_behaviors(_behav_node(0.8))
    adequate, _ = scorer._score_behaviors(_behav_node(1.0))
    excellent, _ = scorer._score_behaviors(_behav_node(1.3))
    assert deficient < adequate < excellent
    assert adequate < 1.0, "meeting target must leave headroom to reward exceeding"
    assert excellent >= 0.99, "clearly exceeding target reaches the top"


def test_perf_score_monotonic_and_bounded():
    prev = -1.0
    for r in [0.0, 0.5, 1.0, 1.15, 1.30, 2.0]:
        s = ScoringAgent._perf_score(r)
        assert 0.0 <= s <= 1.0
        assert s >= prev, "must be non-decreasing in performance ratio"
        prev = s


def _env_node(*, wall_mat, window_ratio, compactness, has_mep, natural=False):
    n = FBSLLayoutNode(); n.layout = Layout()
    r = Room(name="R", room_type="living_room", area=20); n.layout.rooms[r.room_id] = r
    n.layout.compactness_score = compactness
    specs = [
        ("external_wall", StructureType.WALL, wall_mat, "envelope", {'area': 40.0}, True),
        ("roof", StructureType.WALL, wall_mat, "envelope", {'area': 30.0}, True),
        ("window", StructureType.WALL, "glazing", "envelope",
         {'window_ratio': window_ratio, 'area': 8.0}, False),
    ]
    for sname, st, mat, cat, dims, lb in specs:
        s = Structure(name=sname, structure_type=st, material_type=mat, category=cat,
                      dimensions=dict(dims), load_bearing=lb)
        n.structures[s.structure_id] = s
    if has_mep:
        s = Structure(name="hvac", structure_type=StructureType.MEP, material_type="steel",
                      category="services", dimensions={'flow_rate': 0.5})
        n.structures[s.structure_id] = s
    if natural:
        n.metadata['ventilation_strategy'] = 'natural'
    return n


def test_sustainability_is_real_and_ordered():
    green = _env_node(wall_mat="wood", window_ratio=0.18, compactness=0.95,
                      has_mep=False, natural=True)
    standard = _env_node(wall_mat="concrete", window_ratio=0.25, compactness=0.75, has_mep=True)
    poor = _env_node(wall_mat="steel", window_ratio=0.45, compactness=0.40, has_mep=True)

    sg, _ = scorer._score_sustainability(green)
    ss, _ = scorer._score_sustainability(standard)
    sp, _ = scorer._score_sustainability(poor)

    assert len({round(sg, 3), round(ss, 3), round(sp, 3)}) == 3, "must be distinct"
    assert sg > ss > sp, "green > standard > poor"
    assert sg > 0.6 and sp < 0.35, "clear spread, not a flat ~0.5"


def test_sustainability_layout_coupled():
    """Compactness must move S_sust (the form-factor term)."""
    compact = _env_node(wall_mat="concrete", window_ratio=0.2, compactness=0.95, has_mep=True)
    linear = _env_node(wall_mat="concrete", window_ratio=0.2, compactness=0.40, has_mep=True)
    assert scorer._score_sustainability(compact)[0] > scorer._score_sustainability(linear)[0]


def _struct_node(*, wall_mat, wall_lb, wall_thick, foundation, room_w, room_l):
    n = FBSLLayoutNode(); n.layout = Layout()
    r = Room(name="R", room_type="living_room", area=room_w * room_l, width=room_w, length=room_l)
    n.layout.rooms[r.room_id] = r
    s = Structure(name="external_wall", structure_type=StructureType.WALL, material_type=wall_mat,
                  category="envelope", dimensions={'thickness': wall_thick, 'area': 40}, load_bearing=wall_lb)
    n.structures[s.structure_id] = s
    if foundation:
        f = Structure(name="reinforced_concrete_foundation", structure_type=StructureType.FOUNDATION,
                      material_type="concrete", category="structural",
                      dimensions={'thickness': 0.3, 'depth': 0.6}, load_bearing=True)
        n.structures[f.structure_id] = f
    return n


def test_structural_feasibility_is_real_and_ordered():
    feasible = _struct_node(wall_mat="concrete", wall_lb=True, wall_thick=0.2,
                            foundation=True, room_w=4.0, room_l=7.0)
    marginal = _struct_node(wall_mat="concrete", wall_lb=True, wall_thick=0.10,
                            foundation=True, room_w=7.0, room_l=7.0)   # thin + over-span
    infeasible = _struct_node(wall_mat="gypsum_board", wall_lb=True, wall_thick=0.08,
                              foundation=False, room_w=8.0, room_l=8.0)  # bad material, no foundation

    sf, _ = scorer._score_structures(feasible)
    sm, _ = scorer._score_structures(marginal)
    si, _ = scorer._score_structures(infeasible)

    assert sf > sm > si, "feasible > marginal > infeasible"
    assert sf > 0.9 and si < 0.45, "catches genuinely infeasible structure"


def test_structural_invalid_material_penalised():
    """A gypsum (non-structural) load-bearing wall must reduce material validity."""
    good = _struct_node(wall_mat="concrete", wall_lb=True, wall_thick=0.2,
                        foundation=True, room_w=4.0, room_l=5.0)
    bad = _struct_node(wall_mat="gypsum_board", wall_lb=True, wall_thick=0.2,
                       foundation=True, room_w=4.0, room_l=5.0)
    _, dg = scorer._score_structures(good)
    _, db = scorer._score_structures(bad)
    assert dg['material_validity'] == 1.0
    assert db['material_validity'] < 1.0
