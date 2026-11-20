"""Game constants: cards, elixir costs, detection classes.

Extend these lists to match the card pool you train on. Names are kebab-case and
must match folder names under `assets/segments/` and `data/cards/`.
"""

ELIXIR_COST: dict[str, int] = {
    # Every card + evolution with art in assets/card_art (mirror: cost varies, listed as 1).
    "archer-queen": 5, "archers": 3, "archers-evolution": 3, "arrows": 3, "baby-dragon": 4,
    "baby-dragon-evolution": 4, "balloon": 5, "bandit": 3, "barbarian-barrel": 2,
    "barbarian-hut": 6, "barbarians": 5, "barbarians-evolution": 5, "bats": 2,
    "bats-evolution": 2, "battle-healer": 4, "battle-ram": 4, "battle-ram-evolution": 4,
    "berserker": 2, "bomb-tower": 4, "bomber": 2, "bomber-evolution": 2, "boss-bandit": 6,
    "bowler": 5, "cannon": 3, "cannon-cart": 5, "cannon-evolution": 3, "clone": 3,
    "dark-prince": 4, "dart-goblin": 3, "dart-goblin-evolution": 3, "earthquake": 3,
    "electro-dragon": 5, "electro-dragon-evolution": 5, "electro-giant": 7,
    "electro-giant-evolution": 7, "electro-spirit": 1, "electro-wizard": 4,
    "elite-barbarians": 6, "elite-barbarians-evolution": 6, "elixir-collector": 6,
    "elixir-golem": 3, "executioner": 5, "executioner-evolution": 5, "fire-spirit": 1,
    "fireball": 4, "firecracker": 3, "firecracker-evolution": 3, "fisherman": 3,
    "flying-machine": 4, "freeze": 4, "furnace": 4, "furnace-evolution": 4, "giant": 5,
    "giant-evolution": 5, "giant-skeleton": 6, "giant-snowball": 2,
    "giant-snowball-evolution": 2, "goblin-barrel": 3, "goblin-barrel-evolution": 3,
    "goblin-cage": 4, "goblin-cage-evolution": 4, "goblin-curse": 2, "goblin-demolisher": 4,
    "goblin-drill": 4, "goblin-drill-evolution": 4, "goblin-gang": 3, "goblin-giant": 6,
    "goblin-giant-evolution": 6, "goblin-hut": 5, "goblin-machine": 5, "goblins": 2,
    "goblinstein": 5, "golden-knight": 4, "golem": 8, "graveyard": 5, "guards": 3,
    "heal-spirit": 1, "hog-rider": 4, "hunter": 4, "hunter-evolution": 4, "ice-golem": 2,
    "ice-spirit": 1, "ice-spirit-evolution": 1, "ice-wizard": 3, "inferno-dragon": 4,
    "inferno-dragon-evolution": 4, "inferno-tower": 5, "knight": 3, "knight-evolution": 3,
    "lava-hound": 7, "lightning": 6, "little-prince": 3, "lumberjack": 4,
    "lumberjack-evolution": 4, "magic-archer": 4, "mega-knight": 7, "mega-knight-evolution": 7,
    "mega-minion": 3, "mighty-miner": 4, "miner": 3, "mini-pekka": 4, "minion-horde": 5,
    "minion-horde-evolution": 5, "minions": 3, "mirror": 1, "mirror-evolution": 1, "monk": 5,
    "mortar": 4, "mortar-evolution": 4, "mother-witch": 4, "musketeer": 4,
    "musketeer-evolution": 4, "night-witch": 4, "party-hut": 5, "party-rocket": 5, "pekka": 7,
    "pekka-evolution": 7, "phoenix": 4, "poison": 4, "prince": 5, "princess": 3,
    "princess-evolution": 3, "rage": 2, "raging-prince": 5, "ram-rider": 5, "rascals": 5,
    "rocket": 6, "royal-delivery": 3, "royal-ghost": 3, "royal-ghost-evolution": 3,
    "royal-giant": 6, "royal-giant-evolution": 6, "royal-hogs": 5, "royal-hogs-evolution": 5,
    "royal-recruits": 7, "royal-recruits-evolution": 7, "rune-giant": 4, "santa-hog-rider": 5,
    "skeleton-army": 3, "skeleton-army-evolution": 3, "skeleton-barrel": 3,
    "skeleton-barrel-evolution": 3, "skeleton-dragons": 4, "skeleton-king": 4, "skeletons": 1,
    "skeletons-evolution": 1, "sparky": 6, "spear-goblins": 2, "spirit-empress": 3,
    "super-archers": 3, "super-ice-golem": 4, "super-lava-hound": 8, "super-magic-archer": 5,
    "super-mini-pekka": 5, "super-witch": 6, "suspicious-bush": 2, "terry": 4, "tesla": 4,
    "tesla-evolution": 4, "the-log": 2, "three-musketeers": 9, "tombstone": 3, "tornado": 3,
    "valkyrie": 4, "valkyrie-evolution": 4, "vines": 3, "void": 3, "wall-breakers": 2,
    "wall-breakers-evolution": 2, "witch": 5, "witch-evolution": 5, "wizard": 5,
    "wizard-evolution": 5, "x-bow": 6, "zap": 2, "zap-evolution": 2, "zappies": 4,
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
# Every spell card (the rule agent won't use these to defend).
SPELL_CARDS = sorted({
    "arrows", "barbarian-barrel", "clone", "earthquake", "fireball", "freeze", "giant-snowball",
    "goblin-barrel", "goblin-curse", "graveyard", "lightning", "mirror", "party-rocket", "poison",
    "rage", "rocket", "royal-delivery", "the-log", "tornado", "void", "zap",
} | {f"{c}-evolution" for c in ("giant-snowball", "goblin-barrel", "mirror", "zap")})
# Default synthetic-detector pool: the 2.6 hog cycle deck plus common extras.
# (CARD_NAMES covers every card the hand classifier knows; the detector pool stays small.)
_DETECTOR_CARDS = [
    "hog-rider", "musketeer", "cannon", "ice-golem", "skeletons", "ice-spirit", "fireball", "the-log",
    "skeletons-evolution", "ice-spirit-evolution", "knight", "archers", "giant", "mini-pekka",
    "valkyrie", "baby-dragon", "wizard", "goblins", "spear-goblins", "minions", "arrows", "zap",
    "tesla", "inferno-tower", "golem", "pekka", "balloon", "royal-giant", "witch", "bats",
    "goblin-barrel", "princess", "electro-wizard",
]
UNIT_CLASSES = sorted({u for c in _DETECTOR_CARDS for u in CARD_TO_UNITS.get(c, [c])} - set(SPELL_CLASSES))
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
