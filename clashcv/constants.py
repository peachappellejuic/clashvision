"""Game constants: cards, elixir costs, detection classes.

Extend these lists to match the card pool you train on. Names are kebab-case and
must match folder names under `assets/segments/` and `data/cards/`.
"""

ELIXIR_COST: dict[str, int] = {
    # Classic "2.6 Hog Cycle" deck (the deck KataCR's agent plays)
    "hog-rider": 4, "musketeer": 4, "cannon": 3, "ice-golem": 2,
    "skeletons": 1, "ice-spirit": 1, "fireball": 4, "the-log": 2,
    "skeletons-evolution": 1, "ice-spirit-evolution": 1,
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
    "skeletons": ["skeleton"], "skeletons-evolution": ["skeleton-evolution", "skeleton"],
    "ice-spirit-evolution": ["ice-spirit-evolution", "ice-spirit"], "goblins": ["goblin"], "spear-goblins": ["spear-goblin"],
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


# KataCR's detection label list (katacr/constants/label_list.py), in index order.
# Detector classes are these names with a side digit appended: "knight0" = ally,
# "knight1" = enemy (KataCR's "belong" convention).
KATACR_CLASSES: list[str] = """
king-tower queen-tower cannoneer-tower dagger-duchess-tower dagger-duchess-tower-bar tower-bar
king-tower-bar bar bar-level clock emote text elixir selected skeleton-king-bar skeleton
skeleton-evolution electro-spirit fire-spirit ice-spirit heal-spirit goblin spear-goblin bomber bat
bat-evolution zap giant-snowball ice-golem barbarian-barrel barbarian barbarian-evolution
wall-breaker rage the-log archer arrows knight knight-evolution minion cannon skeleton-barrel
firecracker firecracker-evolution royal-delivery royal-recruit royal-recruit-evolution tombstone
mega-minion dart-goblin earthquake elixir-golem-big elixir-golem-mid elixir-golem-small
goblin-barrel guard clone tornado miner dirt princess ice-wizard royal-ghost bandit fisherman
skeleton-dragon mortar mortar-evolution tesla fireball mini-pekka musketeer goblin-cage
goblin-brawler valkyrie battle-ram battle-ram-evolution bomb-tower bomb flying-machine hog-rider
battle-healer furnace zappy baby-dragon dark-prince freeze poison hunter goblin-drill
electro-wizard inferno-dragon phoenix-big phoenix-egg phoenix-small magic-archer lumberjack
night-witch mother-witch hog golden-knight skeleton-king mighty-miner rascal-boy rascal-girl giant
goblin-hut inferno-tower wizard royal-hog witch balloon prince electro-dragon bowler executioner
axe cannon-cart ram-rider graveyard archer-queen monk royal-giant royal-giant-evolution
elite-barbarian rocket barbarian-hut elixir-collector giant-skeleton lightning goblin-giant x-bow
sparky pekka electro-giant mega-knight lava-hound lava-pup golem golemite little-prince
royal-guardian archer-evolution ice-spirit-evolution valkyrie-evolution bomber-evolution
wall-breaker-evolution evolution-symbol mirror tesla-evolution goblin-ball skeleton-king-skill
tesla-evolution-shock ice-spirit-evolution-symbol zap-evolution
""".split()


def split_side(name: str) -> tuple[str, int]:
    """'knight1' -> ('knight', SIDE_ENEMY); names without a side digit -> SIDE_UNKNOWN."""
    if name and name[-1] in "01" and len(name) > 1:
        return name[:-1], int(name[-1])
    return name, SIDE_UNKNOWN
