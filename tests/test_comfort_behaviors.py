"""Comfort behaviour instantiation: a brief that asks for daylight, ventilation,
thermal performance or acoustic separation must actually create those behaviours.

Before this, comfort behaviours were created only by scanning each room's
*per-room* requirement strings for 'light'/'ventilation'. A building-wide
instruction — the ablation study's family-home brief literally says "Prioritise
natural light throughout and good acoustic separation between the bedrooms and
living spaces" — reached no room and created nothing, and thermal and acoustic
behaviours had no creation path anywhere on the live path. The four physics
models in BehaviorCalculator therefore never executed.
"""
from backend.agents.encoder_agent import EncoderAgent as E


def test_building_wide_comfort_language_is_detected():
    """The exact sentence from the family-home brief must register both intents."""
    intents = E._comfort_intents(
        "Prioritise natural light throughout and good acoustic separation "
        "between the bedrooms and living spaces."
    )
    assert intents == {"lighting", "acoustic"}, intents


def test_all_four_intents_detected_together():
    intents = E._comfort_intents(
        "Prioritise abundant natural daylight, cross-ventilation, "
        "energy-efficient passive design, and strong acoustic separation."
    )
    assert intents == {"lighting", "ventilation", "thermal", "acoustic"}, intents


def test_brief_without_comfort_language_creates_nothing():
    """A plain brief must not manufacture comfort requirements the client never
    asked for — the behaviour set should stay purely spatial."""
    assert E._comfort_intents(
        "A small 2-bedroom apartment with one bathroom, a kitchen, and a living room."
    ) == set()


def test_per_room_requirements_still_trigger_intents():
    """Room-level phrasing keeps working alongside the brief-level scan."""
    assert "acoustic" in E._comfort_intents("quiet home office")
    assert "lighting" in E._comfort_intents("large windows facing south")


def test_comfort_behaviours_apply_only_to_sensible_room_types():
    """A garage does not need a daylight factor; a closet does not need an
    acoustic target. Applicability is what stops the behaviour set filling with
    requirements no one would design to."""
    applies = E._COMFORT_APPLIES
    assert "garage" not in applies["lighting"]
    assert "storage" not in applies["lighting"]
    assert "bedroom" in applies["lighting"]
    # wet rooms need extract ventilation but not a daylight target
    assert "bathroom" in applies["ventilation"]
    assert "bathroom" not in applies["lighting"]
    # acoustic covers both the quiet rooms and the noise sources
    assert {"bedroom", "office", "living_room"} <= applies["acoustic"]


def test_every_intent_has_an_applicability_rule():
    """A cue with no applicability entry would silently never create anything."""
    assert set(E._COMFORT_CUES) == set(E._COMFORT_APPLIES)
