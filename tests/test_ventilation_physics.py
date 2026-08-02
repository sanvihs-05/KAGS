"""Ventilation behaviour: air changes per hour computed from real opening
geometry (BS 5925 / CIBSE AM10 concept-stage envelope flow), not a strategy
label lookup.

The previous implementation scored a design by which objects existed — HVAC
present → 1.0, windows → 0.75, nothing → 0.40 — so it could not distinguish two
naturally ventilated designs with very different glazing, and gave any design
carrying an HVAC object a perfect score regardless of capacity. These tests pin
the properties that made the replacement worth making.
"""
from types import SimpleNamespace as NS

from backend.core.behavior_calculator import BehaviorCalculator

ROOMS = (("bedroom", 16.0), ("living_room", 40.0), ("closet", 6.0))


def _case(glazed_types, ratio=0.18, hvac=True, rooms=ROOMS):
    """Build the minimal duck-typed node/structures the calculator reads."""
    node = NS(layout=NS(rooms={
        f"r{i}": NS(area=a, height=3.0, room_type=t) for i, (t, a) in enumerate(rooms)
    }))
    structures = {
        t: NS(name=f"{t}_window", dimensions={"window_ratio": ratio},
              structure_type=NS(value="wall"))
        for t in glazed_types
    }
    if hvac:
        structures["hvac"] = NS(name="hvac_ventilation_system", dimensions={},
                                structure_type=NS(value="mep"))
    return node, structures


def _ach(glazed_types, **kw):
    bc = BehaviorCalculator()
    node, structures = _case(glazed_types, **kw)
    return bc._calculate_ventilation_behavior(NS(target_value=None), structures, node)


def test_ach_increases_with_glazing_ratio():
    """More openable area must mean more air. This is the discrimination the
    label lookup could not provide."""
    all_types = tuple(t for t, _ in ROOMS)
    rates = [_ach(all_types, ratio=r, hvac=False) for r in (0.05, 0.10, 0.18, 0.25)]
    assert rates == sorted(rates), rates
    assert rates[0] < rates[-1] / 2, "glazing should change the rate substantially"


def test_open_window_rates_are_physically_plausible():
    """Openable windows give purge-scale ventilation — single-digit ACH, not the
    ~0.5 ACH of background/trickle ventilation, and not an absurd number."""
    ach = _ach(tuple(t for t, _ in ROOMS), ratio=0.18, hvac=False)
    assert 3.0 < ach < 15.0, ach


def test_windowless_room_lowers_the_building_rate():
    """An interior room served only by plant must drag the dwelling figure down;
    the floor-area weighting is what makes deep plans score worse."""
    all_types = tuple(t for t, _ in ROOMS)
    assert _ach(all_types) > _ach(("bedroom", "living_room"))


def test_sealed_design_falls_back_to_infiltration_not_zero():
    """No openings and no plant is still not a vacuum flask — envelope leakage
    gives a small non-zero rate."""
    ach = _ach((), ratio=0.0, hvac=False)
    assert 0.0 < ach < 0.5, ach


def test_hvac_is_not_a_free_perfect_score():
    """The old lookup returned 1.0 for anything with an HVAC object. A sealed
    mechanically ventilated design must now sit far below an openable-window
    design on the purge criterion it is scored against."""
    sealed_mech = _ach((), ratio=0.0, hvac=True)
    natural = _ach(tuple(t for t, _ in ROOMS), ratio=0.18, hvac=False)
    assert sealed_mech < natural
    assert sealed_mech > _ach((), ratio=0.0, hvac=False), "plant must still help"


def test_performance_ratio_is_scaled_against_the_target():
    """With a target set the calculator returns an actual_value on the target's
    scale (target x performance ratio), matching the other behaviour models."""
    bc = BehaviorCalculator()
    node, structures = _case(tuple(t for t, _ in ROOMS), ratio=0.10, hvac=True)
    actual = bc._calculate_ventilation_behavior(NS(target_value=4.0), structures, node)
    bare = bc._calculate_ventilation_behavior(NS(target_value=None), structures, node)
    assert abs(actual - 4.0 * min(2.0, bare / bc._VENT_PURGE_TARGET)) < 1e-9


def test_missing_geometry_does_not_crash():
    """No layout rooms — fall back to a presence heuristic rather than raising."""
    bc = BehaviorCalculator()
    node = NS(layout=None)
    structures = {"w": NS(name="bedroom_window", dimensions={"window_ratio": 0.2},
                          structure_type=NS(value="wall"))}
    assert bc._calculate_ventilation_behavior(NS(target_value=None), structures, node) > 0
