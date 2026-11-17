import numpy as np

from clashcv.generation import SyntheticGenerator, grabcut_rgba
from clashcv.policy import AdbController, RuleAgent
from clashcv.perception.sar import GameState, Unit
from clashcv.constants import SIDE_ENEMY

from conftest import sprite


def _gen(seed=0, **kw):
    segs = {"knight": [sprite((200, 50, 50))], "archer": [sprite((50, 200, 50), (30, 50))],
            "king-tower": [sprite((120, 120, 120), (120, 120))],
            "queen-tower": [sprite((150, 150, 150), (90, 100))]}
    bg = [np.full((896, 576, 3), (60, 110, 70), np.uint8)]
    return SyntheticGenerator(segs, bg, seed=seed, **kw)


def test_generator_labels_are_valid():
    g = _gen()
    for _ in range(10):
        img, labels = g.sample()
        assert img.shape == (896, 576, 3)
        assert labels, "expected some objects"
        for c, x, y, w, h in labels:
            assert 0 <= c < len(g.names)
            assert 0 <= x - w / 2 + 1e-6 and x + w / 2 <= 1 + 1e-6
            assert 0 <= y - h / 2 + 1e-6 and y + h / 2 <= 1 + 1e-6


def test_generator_drops_occluded():
    bg = [np.full((896, 576, 3), (60, 110, 70), np.uint8)]
    g = SyntheticGenerator({"knight": [sprite((200, 50, 50))]}, bg, units_range=(3, 3), seed=0)
    g.rng.uniform = lambda a, b: (a + b) / 2  # every unit lands on the same spot, same size
    _, labels = g.sample()
    assert len(labels) == 1  # the two fully hidden copies are not labelled


def test_write_dataset(tmp_path):
    y = _gen().write_dataset(tmp_path, 3, 2)
    assert y.exists()
    assert len(list((tmp_path / "images/train").iterdir())) == 3
    assert len(list((tmp_path / "labels/val").iterdir())) == 2


def test_grabcut():
    crop = np.full((80, 60, 3), (60, 110, 70), np.uint8)
    crop[15:65, 15:45] = (20, 20, 220)
    rgba = grabcut_rgba(crop)
    assert rgba is not None and rgba.shape == (80, 60, 4)
    assert rgba[40, 30, 3] == 255 and rgba[2, 2, 3] == 0


def test_rule_agent_defends(cfg):
    st = GameState(100, 120, 6.0, ["hog-rider", "skeletons", "fireball", "cannon"], [1] * 4,
                   [Unit("knight", SIDE_ENEMY, (5.0, 22.0), (0, 0, 1, 1), 1, 0.9)], {})
    move = RuleAgent().act(st)
    assert move is not None
    slot, tile = move
    assert st.hand[slot] == "skeletons"  # cheapest troop
    assert tile[1] > 22
    ctrl = AdbController(cfg, 1080, 2400, dry_run=True)
    ctrl.play(slot, tile)
    assert len(ctrl.log) == 2
