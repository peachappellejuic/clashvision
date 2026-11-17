"""Game constants: cards, elixir costs, detection classes.

Extend these lists to match the card pool you train on. Names are kebab-case and
must match folder names under `assets/segments/` and `data/cards/`.
"""

ELIXIR_COST: dict[str, int] = {
    # Classic "2.6 Hog Cycle" deck (the deck KataCR's agent plays)
    "hog-rider": 4, "musketeer": 4, "cannon": 3, "ice-golem": 2,
    "skeletons": 1, "ice-spirit": 1, "fireball": 4, "the-log": 2,
    # Common extras
    "knight": 3, "archers": 3, "giant": 5, "mini-pekka": 4, "valkyrie": 4,
    "baby-dragon": 4, "wizard": 5, "goblins": 2, "spear-goblins": 2,
    "minions": 3, "arrows": 3, "zap": 2, "tesla": 4, "inferno-tower": 5,
    "golem": 8, "pekka": 7, "balloon": 5, "royal-giant": 6, "witch": 5,
    "bats": 2, "goblin-barrel": 3, "princess": 3, "electro-wizard": 4,
}

CARD_NAMES: list[str] = sorted(ELIXIR_COST)
EMPTY_CARD = "empty"  # hand slot while a card is cycling in

# Card -> unit classes it spawns on the arena (used to localise played cards).
CARD_TO_UNITS: dict[str, list[str]] = {
    "skeletons": ["skeleton"], "goblins": ["goblin"], "spear-goblins": ["spear-goblin"],
    "minions": ["minion"], "archers": ["archer"], "bats": ["bat"],
    "goblin-barrel": ["goblin", "goblin-barrel"],
    "fireball": ["fireball"], "the-log": ["the-log"], "arrows": ["arrows"], "zap": ["zap"],
}

TOWER_CLASSES = ["king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower"]
SPELL_CLASSES = ["fireball", "the-log", "arrows", "zap", "goblin-barrel"]
UNIT_CLASSES = sorted({u for c in CARD_NAMES for u in CARD_TO_UNITS.get(c, [c])} - set(SPELL_CLASSES))
# Detector classes: units + spells + towers + misc UI elements (bars, clock, ...).
DETECTION_CLASSES: list[str] = UNIT_CLASSES + SPELL_CLASSES + TOWER_CLASSES + ["bar", "clock", "emote"]

SIDE_ALLY, SIDE_ENEMY, SIDE_UNKNOWN = 0, 1, -1


def units_for_card(card: str) -> list[str]:
    return CARD_TO_UNITS.get(card, [card])
