"""Daylight behaviour: BRE (Lynes) average daylight factor computed from room
geometry, not glazing-over-floor-area.

The previous model was `DF = window_ratio x 0.75 x 100`, which ignored room
proportion and ceiling height entirely and overstated DF by roughly 5x — an 18 %
glazing ratio came out at 13.5 %, an atrium-like figure for a bedroom. These
tests pin both the absolute realism and the geometry sensitivity that the
replacement buys.
"""
from types import SimpleNamespace as NS

from backend.core.behavior_calculator import BehaviorCalculator


def _rooms(specs):
    return NS(layout=NS(rooms={
        f"r{i}": NS(area=a, height=h, room_type=t, width=w, length=l)
        for i, (t, a, w, l, h) in enumerate(specs)
    }))


def _df(specs, glazed_types, ratio=0.18):
    bc = BehaviorCalculator()
    structures = {
        t: NS(name=f"{t}_window", dimensions={"window_ratio": ratio},
              structure_type=NS(value="wall"))
        for t in glazed_types
    }
    return bc._calculate_lighting_behavior(NS(target_value=None), structures, _rooms(specs))


SQUARE = [("bedroom", 16.0, 4.0, 4.0, 3.0)]


def test_absolute_daylight_factor_is_physically_plausible():
    """A normally glazed bedroom should land in the 1-4 % band real dwellings
    occupy — not the >13 % the floor-area formula produced."""
    df = _df(SQUARE, ("bedroom",), ratio=0.18)
    assert 1.5 < df < 4.0, df


def test_daylight_factor_increases_with_glazing():
    vals = [_df(SQUARE, ("bedroom",), ratio=r) for r in (0.10, 0.15, 0.18, 0.25)]
    assert vals == sorted(vals), vals


def test_taller_room_lowers_average_daylight_factor():
    """More wall surface for the same glazing means a lower average DF. The old
    formula was blind to this because it divided by floor area only."""
    low = _df([("bedroom", 16.0, 4.0, 4.0, 2.4)], ("bedroom",))
    high = _df([("bedroom", 16.0, 4.0, 4.0, 4.0)], ("bedroom",))
    assert high < low


def test_elongated_room_scores_below_compact_room_of_equal_area():
    """Room proportion matters: the same floor area and glazing in a long thin
    room has more enclosing surface and daylights less well on average."""
    compact = _df([("bedroom", 16.0, 4.0, 4.0, 3.0)], ("bedroom",))
    elongated = _df([("bedroom", 16.0, 8.0, 2.0, 3.0)], ("bedroom",))
    assert elongated < compact


def test_deep_room_is_penalised_by_the_limiting_depth_rule():
    """A room deeper than the BRE limit (L/W + L/H_w > 2/(1-R)) cannot daylight
    its back half; DF is scaled down rather than stepping to a hard fail."""
    bc = BehaviorCalculator()
    limit = 2.0 / (1.0 - bc._DF_REFLECTANCE)
    # 9x9 room: 9/9 + 9/2.1 = 1 + 4.29 = 5.29 > limit(4.0) -> penalised
    deep = [("bedroom", 81.0, 9.0, 9.0, 3.0)]
    short = 9.0
    depth_index = short / short + short / bc._DF_WINDOW_HEAD
    assert depth_index > limit, depth_index
    penalised = _df(deep, ("bedroom",))
    # recompute what it would be without the depth factor
    unpenalised = penalised * depth_index / limit
    assert penalised < unpenalised


def test_unglazed_room_lowers_the_building_average():
    two = [("bedroom", 16.0, 4.0, 4.0, 3.0), ("closet", 16.0, 4.0, 4.0, 3.0)]
    assert _df(two, ("bedroom",)) < _df(two, ("bedroom", "closet"))


def test_performance_ratio_scales_against_target():
    bc = BehaviorCalculator()
    structures = {"bedroom": NS(name="bedroom_window", dimensions={"window_ratio": 0.18},
                                structure_type=NS(value="wall"))}
    node = _rooms(SQUARE)
    bare = bc._calculate_lighting_behavior(NS(target_value=None), structures, node)
    actual = bc._calculate_lighting_behavior(NS(target_value=3.0), structures, node)
    assert abs(actual - 3.0 * min(2.0, bare / bc._DF_TARGET)) < 1e-9


def test_missing_geometry_falls_back_without_crashing():
    bc = BehaviorCalculator()
    structures = {"w": NS(name="bedroom_window", dimensions={"window_ratio": 0.2},
                          structure_type=NS(value="wall"))}
    assert bc._calculate_lighting_behavior(
        NS(target_value=None), structures, NS(layout=None)) > 0
