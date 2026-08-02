"""Thermal behaviour and the opaque envelope.

Two defects are pinned here.

1. The encoder created no exterior wall or roof, so the only structures the
   thermal model recognised as envelope were the windows and the foundation
   slab. It averaged glazing against bare concrete and returned R ~ 0.4 against
   a target of 5.0 for every design — a constant penalty describing the model
   rather than the design.
2. No variant strategy touched the envelope, so a strategy whose stated purpose
   is exceeding thermal targets produced exactly the same R-value as one whose
   purpose is minimising material, and thermal contributed nothing to ranking.
"""
from types import SimpleNamespace as NS

from backend.core.behavior_calculator import BehaviorCalculator

FLOOR_AREA = 250.0
WALL_AREA = 4.0 * (FLOOR_AREA ** 0.5) * 3.0


def _envelope(wall_material, roof_material, with_windows=True):
    structures = {
        "wall": NS(name="exterior_wall", material_type=wall_material, category="envelope",
                   dimensions={"area": WALL_AREA, "thickness": 0.30}, acoustic_rating=None),
        "roof": NS(name="roof", material_type=roof_material, category="envelope",
                   dimensions={"area": FLOOR_AREA, "thickness": 0.40}, acoustic_rating=None),
        "found": NS(name="reinforced_concrete_foundation", material_type="concrete",
                    category="structural", dimensions={"thickness": 0.3}, acoustic_rating=None),
    }
    if with_windows:
        for i in range(14):
            structures[f"w{i}"] = NS(name="bedroom_window", material_type="glazing",
                                     category="envelope",
                                     dimensions={"window_ratio": 0.18}, acoustic_rating=None)
    return structures


def _r_value(wall_material, roof_material, **kw):
    """actual = target x (R/5.0) with target 5.0, so actual is R itself."""
    bc = BehaviorCalculator()
    return bc._calculate_thermal_behavior(
        NS(target_value=5.0), _envelope(wall_material, roof_material, **kw), NS(layout=None))


def test_envelope_spec_orders_thermal_performance():
    """The three build-ups the strategies choose between must rank correctly:
    lightweight < default < high-performance."""
    light = _r_value("lightweight_frame", "lightweight_roof")
    default = _r_value("insulated_timber_frame", "insulated_roof")
    high = _r_value("high_performance_envelope", "high_performance_roof")
    assert light < default < high, (light, default, high)


def test_insulated_envelope_reaches_a_realistic_r_value():
    """A modern insulated envelope should land near the R 5.0 target, not the
    R ~0.4 produced when only glazing and a concrete slab were visible."""
    r = _r_value("insulated_timber_frame", "insulated_roof")
    assert 4.0 < r < 8.0, r


def test_lightweight_envelope_falls_below_target():
    """The material-minimising strategy should genuinely underperform on
    thermal — that trade-off is the point of having the strategy."""
    assert _r_value("lightweight_frame", "lightweight_roof") < 5.0


def test_bare_material_envelope_scores_far_below_an_assembly():
    """Bare-material entries describe uninsulated elements; they must not be
    mistaken for build-ups. This is what the old envelope-less design measured."""
    bare = _r_value("concrete", "concrete")
    assembly = _r_value("insulated_timber_frame", "insulated_roof")
    assert bare < 1.0 < assembly


def test_opaque_envelope_dominates_the_glazing_in_the_area_weighting():
    """Windows carry a nominal unit area, so adding them must not swamp the
    wall and roof — otherwise the average tracks the glass, not the envelope."""
    with_win = _r_value("insulated_timber_frame", "insulated_roof", with_windows=True)
    without = _r_value("insulated_timber_frame", "insulated_roof", with_windows=False)
    assert abs(with_win - without) < 0.5, (with_win, without)


def test_missing_envelope_falls_back_without_crashing():
    bc = BehaviorCalculator()
    assert bc._calculate_thermal_behavior(NS(target_value=5.0), {}, NS(layout=None)) > 0


def test_refinement_materially_improves_thermal():
    """Type-1 reformulation used to append a `thermal_insulation` element with no
    `area`, so the area-weighted average gave it a weight of 1 against a ~190 m²
    wall and a 250 m² roof — the loop believed it had addressed the deviation
    while the score moved by well under a percent. Upgrading the build-up has to
    show up."""
    from backend.agents.refinement_agent import RefinementAgent
    from backend.core.fbsl_models import (FBSLLayoutNode, Layout, Room,
                                          Structure, StructureType)

    node = FBSLLayoutNode()
    node.layout = Layout()
    room = Room(name="Living", room_type="living_room", area=250.0)
    node.layout.rooms = {room.room_id: room}
    for name, material, area in (("exterior_wall", "insulated_timber_frame", 190.0),
                                 ("roof", "insulated_roof", 250.0)):
        node.add_structure(Structure(name=name, structure_type=StructureType.WALL,
                                     material_type=material, category="envelope",
                                     dimensions={"area": area, "thickness": 0.3}))

    bc = BehaviorCalculator()
    before = bc._calculate_thermal_behavior(NS(target_value=5.0), node.structures, node)
    RefinementAgent()._add_thermal_structure(node)
    after = bc._calculate_thermal_behavior(NS(target_value=5.0), node.structures, node)

    assert after - before > 1.0, (before, after)
    assert {s.material_type for s in node.structures.values()} == {
        "high_performance_envelope", "high_performance_roof"}


def test_refinement_without_an_envelope_adds_a_weighted_layer():
    """Fallback path: an insulation layer must carry an area, or the calculator
    weights it at 1.0 and effectively ignores it."""
    from backend.agents.refinement_agent import RefinementAgent
    from backend.core.fbsl_models import FBSLLayoutNode, Layout, Room

    node = FBSLLayoutNode()
    node.layout = Layout()
    room = Room(name="Living", room_type="living_room", area=100.0)
    node.layout.rooms = {room.room_id: room}
    RefinementAgent()._add_thermal_structure(node)
    added = [s for s in node.structures.values() if s.name == "thermal_insulation"]
    assert added and (added[0].dimensions or {}).get("area", 0) > 1.0
