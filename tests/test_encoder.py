"""Encoder rule-based parser: room extraction, per-room areas, adjacency
phrases, the extended lexicon (sauna/closet/entry/balcony), and the guards
against real-LLM quirks (literal area 0, name-vs-type adjacency labels)."""
from collections import Counter
from backend.agents.encoder_agent import EncoderAgent
from backend.core.fbsl_models import Room

# skip the heavy __init__ (LLM probe, embeddings) — we only test pure parsing
enc = object.__new__(EncoderAgent)


def _types(parsed):
    return Counter(r['type'] for r in parsed['rooms'])


def test_fallback_extracts_rooms_and_headline_count():
    p = enc._fallback_parse(
        "A 4-bedroom home with a kitchen, living room, two bathrooms and a garage.")
    t = _types(p)
    assert t['bedroom'] == 4          # headline count authoritative
    assert t['bathroom'] == 2
    assert t['kitchen'] == 1 and t['garage'] == 1


def test_fallback_extracts_adjacencies():
    p = enc._fallback_parse(
        "A home with a kitchen connected to the dining area, a master bedroom "
        "with attached bathroom, and a mudroom that connects to the garage.")
    adj = {tuple(sorted((a['room1'], a['room2']))) for a in p['adjacencies']}
    assert ('dining', 'kitchen') in adj
    assert ('bathroom', 'bedroom') in adj
    assert ('garage', 'mudroom') in adj


def test_new_room_types_recognised():
    p = enc._fallback_parse(
        "A home with a sauna, a walk-in closet, an entry hall, and a balcony.")
    t = _types(p)
    for rt in ('sauna', 'closet', 'entry', 'balcony'):
        assert t[rt] == 1, f"{rt} not extracted"


def test_closet_distinct_from_storage():
    p = enc._fallback_parse("An apartment with a storage room and a closet.")
    t = _types(p)
    assert t['closet'] == 1 and t['storage'] == 1


def test_zero_area_from_llm_is_defaulted():
    program = {'rooms': [
        {'type': 'bedroom', 'name': 'Master', 'area_min': 16, 'area_max': 16},
        {'type': 'kitchen', 'name': 'Kitchen', 'area_min': 0, 'area_max': 0},
    ], 'adjacencies': [], 'constraints': [], 'priorities': []}
    out = enc._validate_spatial_program(program)
    by = {r['type']: r for r in out['rooms']}
    assert by['bedroom']['area_min'] == 16.0        # real value preserved
    assert by['kitchen']['area_min'] > 0            # literal 0 replaced with default
    assert by['kitchen']['area_max'] > 0


def test_parse_total_area():
    assert EncoderAgent._parse_total_area("Total area 210-250 sqm.") == (210.0, 250.0)
    lo, hi = EncoderAgent._parse_total_area("a home of about 200 square metres")
    assert lo < 200 < hi
    # a single-room area must NOT be read as the whole-design total
    assert EncoderAgent._parse_total_area("master bedroom 18 sqm") is None


def test_fit_rooms_to_stated_total():
    """A room program summing below the stated total is scaled up to reach it
    (clamped to per-room max)."""
    from backend.core.fbsl_models import FBSLLayoutNode, Function, Room, Layout, FunctionCategory
    n = FBSLLayoutNode(); n.layout = Layout()
    for rt, a in [("bedroom", 14), ("bedroom", 14), ("kitchen", 16), ("living_room", 30)]:
        f = Function(name=f"provide_{rt}", category=FunctionCategory.SPATIAL, priority=0.8,
                     spatial_requirements={'min_area': a * 0.7, 'max_area': a * 1.5, 'preferred_area': a})
        n.functions[f.function_id] = f
        r = Room(name=rt, room_type=rt, area=a, function_id=f.function_id)
        n.layout.rooms[r.room_id] = r
    n.layout.total_area = 74
    EncoderAgent._fit_rooms_to_total(n, (110.0, 130.0))
    total = sum(r.area for r in n.layout.rooms.values())
    assert total >= 110 * 0.98, f"program should reach the stated band minimum, got {total}"


def test_adjacency_labels_resolved_to_types():
    rooms = {}
    for rid, name, rtype in [('r1', 'Master Bedroom', 'bedroom'),
                             ('r2', 'Ensuite Bathroom', 'bathroom'),
                             ('r3', 'Kitchen', 'kitchen')]:
        rooms[rid] = Room(room_id=rid, name=name, room_type=rtype, area=10)
    raw = [
        {'room1': 'Master Bedroom', 'room2': 'Ensuite Bathroom', 'type': 'required'},  # names
        {'room1': 'Nonexistent', 'room2': 'kitchen', 'type': 'required'},              # unresolvable
    ]
    resolved = EncoderAgent._resolve_adjacency_labels(raw, rooms)
    pairs = {(r['room1'], r['room2']) for r in resolved}
    assert ('bedroom', 'bathroom') in pairs
    assert len(resolved) == 1, "unresolvable pair dropped"
