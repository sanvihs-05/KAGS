"""Room proportions when fitting a stated total, and what the drawings label.

Two defects a reader spots immediately on a generated plan:

1. Bathrooms came out the same size as bedrooms. Scaling the programme up to
   reach a stated total ("180-210 sqm") multiplied every room by the same
   factor, clamped only by whatever max_area the LLM had proposed — and LLMs
   propose generous bathroom bands. A bathroom grew 6.0 -> 10.2 m² beside a
   bedroom capped at 16.0.
2. Four rooms the brief had named — master suite, nursery, children's bedroom,
   guest bedroom — all drew as "Bedroom", because every render site labelled by
   `room_type` instead of the room's name.
"""
from backend.agents.encoder_agent import EncoderAgent
from backend.core.fbsl_models import (FBSLLayoutNode, Layout, Room, Function,
                                      FunctionCategory)
from backend.visualization.enhanced_layout import room_label


def _node(spec):
    """spec: [(name, room_type, preferred, min, max)]"""
    node = FBSLLayoutNode()
    node.layout = Layout()
    for name, rtype, pref, lo, hi in spec:
        func = Function(name=f"provide_{rtype}", category=FunctionCategory.SPATIAL,
                        priority=0.9,
                        spatial_requirements={'min_area': lo, 'preferred_area': pref,
                                              'max_area': hi})
        node.add_function(func)
        room = Room(name=name, room_type=rtype, area=pref, function_id=func.function_id)
        node.layout.rooms[room.room_id] = room
    return node


# a generous LLM bathroom band is the trigger, so the fixture uses one
PROGRAMME = [
    ("Ensuite Bathroom", "bathroom", 6.0, 4.0, 12.0),
    ("Children's Bathroom", "bathroom", 6.0, 4.0, 12.0),
    ("Master Bedroom", "bedroom", 14.0, 12.0, 16.0),
    ("Family Room", "living_room", 30.0, 25.0, 40.0),
]


def _areas_by_type(node):
    out = {}
    for r in node.layout.rooms.values():
        out.setdefault(r.room_type, []).append(r.area)
    return out


def test_scaling_up_does_not_inflate_a_bathroom_to_bedroom_size():
    node = _node(PROGRAMME)
    EncoderAgent._fit_rooms_to_total(node, (95.0, 105.0))
    areas = _areas_by_type(node)
    assert max(areas["bathroom"]) < max(areas["bedroom"]) * 0.6, areas


def test_service_rooms_are_capped_at_their_typology_band():
    """A bathroom is sized by what happens in it, not by how big the house is,
    so the typology ceiling overrides a generous LLM max_area."""
    node = _node(PROGRAMME)
    EncoderAgent._fit_rooms_to_total(node, (95.0, 105.0))
    ceiling = EncoderAgent._DEFAULT_AREA_BAND["bathroom"][1]
    assert all(a <= ceiling + 1e-6 for a in _areas_by_type(node)["bathroom"])


def test_habitable_rooms_still_absorb_the_surplus():
    """Capping the wet rooms must not stop the design growing — the living space
    should take the space instead."""
    node = _node(PROGRAMME)
    before = {r.name: r.area for r in node.layout.rooms.values()}
    EncoderAgent._fit_rooms_to_total(node, (95.0, 105.0))
    after = {r.name: r.area for r in node.layout.rooms.values()}
    assert after["Family Room"] > before["Family Room"]
    assert after["Master Bedroom"] > before["Master Bedroom"]


def test_an_unreachable_total_leaves_an_honest_shortfall():
    """When every room is at its ceiling the programme simply cannot reach the
    band. It must stop there rather than inflate service rooms to fake it."""
    node = _node(PROGRAMME)
    EncoderAgent._fit_rooms_to_total(node, (95.0, 105.0))
    total = sum(r.area for r in node.layout.rooms.values())
    ceiling = EncoderAgent._DEFAULT_AREA_BAND["bathroom"][1]
    assert total < 95.0                                    # shortfall, not faked
    assert all(a <= ceiling + 1e-6 for a in _areas_by_type(node)["bathroom"])


def test_scaling_down_is_unaffected_by_the_service_cap():
    """Over-shoot still shrinks toward the band maximum."""
    node = _node([("Family Room", "living_room", 80.0, 25.0, 90.0),
                  ("Master Bedroom", "bedroom", 40.0, 12.0, 45.0)])
    EncoderAgent._fit_rooms_to_total(node, (60.0, 70.0))
    assert sum(r.area for r in node.layout.rooms.values()) <= 70.0 + 1e-6


# ------------------------------------------------------------------ labels
def test_drawings_use_the_brief_given_room_name():
    assert room_label({"name": "Master Bedroom", "room_type": "bedroom"}) == "Master Bedroom"
    assert room_label({"name": "Nursery", "room_type": "bedroom"}) == "Nursery"


def test_rooms_of_the_same_type_stay_distinguishable():
    """The actual complaint: four bedrooms must not all read 'Bedroom'."""
    rooms = [{"name": n, "room_type": "bedroom"} for n in
             ("Master Bedroom", "Nursery", "Children's Bedroom", "Guest Bedroom")]
    assert len({room_label(r) for r in rooms}) == 4


def test_label_falls_back_to_the_type_when_unnamed():
    assert room_label({"name": "", "room_type": "living_room"}) == "Living Room"
    assert room_label({"room_type": "kitchen"}) == "Kitchen"
    assert room_label({}) == "Space"


# --------------------------------------------------- stated-total parsing
def test_dwelling_types_beyond_house_and_home_are_recognised():
    """A brief opening "a bungalow of 180-210 sqm" was not matched at all, so
    the parser fell through and read a *room* size as the building total."""
    assert EncoderAgent._parse_total_area("a bungalow of 180-210 sqm") == (180.0, 210.0)
    for phrase, expected in (("a villa of 200 sqm", 200.0),
                             ("a cottage of 150 sqm", 150.0),
                             ("a duplex of 240 sqm", 240.0)):
        band = EncoderAgent._parse_total_area(phrase)
        assert band and abs(sum(band) / 2 - expected) < 1e-6, (phrase, band)


def test_a_room_size_is_never_taken_as_the_building_total():
    """The structural guard word lists cannot provide: a figure smaller than a
    room already extracted cannot be the total, whatever phrasing produced it."""
    node = _node([("Master Bedroom", "bedroom", 24.0, 20.0, 26.0)])
    assert EncoderAgent._plausible_total((15.0, 21.0), node) is False
    assert EncoderAgent._plausible_total((180.0, 210.0), node) is True


def test_briefs_stating_no_total_are_unaffected():
    assert EncoderAgent._parse_total_area("A small 2-bedroom apartment.") is None
