import numpy as np
import pytest

from clashcv.classification import TemplateCardClassifier
from clashcv.config import ScreenConfig
from clashcv.constants import SIDE_ALLY, SIDE_ENEMY
from clashcv.detection import Detection, IoUTracker, infer_side, nms
from clashcv.perception import Perceiver, parse_clock, read_elixir
from clashcv.regions import Grid, crop, to_pixels

from conftest import H, W, card_image, fake_frame


@pytest.mark.parametrize("e", [0, 2, 5, 7.5, 10])
def test_elixir(cfg, e):
    assert read_elixir(fake_frame(cfg, e), cfg) == pytest.approx(e, abs=0.25)


def test_grid_roundtrip(cfg):
    g = Grid(cfg, W, H)
    for c, r in [(0, 0), (17, 31), (9, 16)]:
        tx, ty = g.pixel_to_tile(*g.tile_center(c, r))
        assert (int(tx), int(ty)) == (c, r)


def test_config_yaml_roundtrip(tmp_path, cfg):
    cfg.timer = (0.1, 0.2, 0.3, 0.4)
    cfg.save(tmp_path / "c.yaml")
    assert ScreenConfig.load(tmp_path / "c.yaml") == cfg


def test_nms_and_tracker():
    a = Detection((0, 0, 10, 10), 0.9, "knight")
    b = Detection((1, 1, 11, 11), 0.8, "knight")
    c = Detection((1, 1, 11, 11), 0.8, "archer")
    assert len(nms([a, b, c])) == 2
    tr = IoUTracker()
    d1 = tr.update([Detection((0, 0, 10, 10), 0.9, "knight", 0)])
    d2 = tr.update([Detection((2, 0, 12, 10), 0.9, "knight", 0), Detection((50, 50, 60, 60), 0.9, "knight", 0)])
    assert d2[0].track_id == d1[0].track_id and d2[1].track_id != d1[0].track_id


def test_side_from_bar_color():
    img = np.full((400, 400, 3), (60, 110, 70), np.uint8)
    img[95:105, 100:150] = (255, 120, 20)  # blue bar (BGR) above unit
    img[95:105, 250:300] = (20, 20, 230)   # red bar
    ally = Detection((100, 105, 150, 180), 0.9, "knight")
    enemy = Detection((250, 105, 300, 180), 0.9, "knight")
    assert infer_side(img, ally) == SIDE_ALLY
    assert infer_side(img, enemy) == SIDE_ENEMY


def test_parse_clock():
    assert parse_clock("2:05") == 125
    assert parse_clock("0.59") == 59
    assert parse_clock("") is None


def _templates(tmp_path, cfg, names):
    import cv2

    for n in names:
        x0, y0, x1, y1 = to_pixels(cfg.cards[0], W, H)
        cv2.imwrite(str(tmp_path / f"{n}.png"), card_image(n, (x1 - x0, y1 - y0)))
    return TemplateCardClassifier(tmp_path)


def test_template_cards(tmp_path, cfg):
    hand = ["hog-rider", "musketeer", "cannon", "ice-golem"]
    clf = _templates(tmp_path, cfg, hand + ["fireball", "the-log"])
    frame = fake_frame(cfg, 5, hand)
    assert [p[0] for p in clf.predict([crop(frame, b) for b in cfg.cards])] == hand


def test_action_detection_and_localisation(tmp_path, cfg):
    """Play hog-rider from slot 0: hand changes, elixir drops 4, a new ally hog appears."""
    hand1 = ["hog-rider", "musketeer", "cannon", "ice-golem"]
    hand2 = ["fireball", "musketeer", "cannon", "ice-golem"]
    clf = _templates(tmp_path, cfg, hand1 + ["fireball"])
    per = Perceiver(cfg, card_classifier=clf)
    g = Grid(cfg, W, H)
    px, py = g.tile_center(4, 18)
    hog = Detection((px - 30, py - 80, px + 30, py), 0.9, "hog-rider", SIDE_ALLY)
    knight = Detection((500, 700, 560, 780), 0.9, "knight", SIDE_ENEMY)

    _, acts, _ = per.step(fake_frame(cfg, 7, hand1), [knight])
    assert acts == []
    _, acts, _ = per.step(fake_frame(cfg, 3, hand2), [knight])
    assert acts == []  # pending until the unit shows up
    _, acts, _ = per.step(fake_frame(cfg, 3.1, hand2), [knight, hog])
    assert len(acts) == 1
    a = acts[0]
    assert (a.card, a.slot) == ("hog-rider", 0)
    assert abs(a.tile[0] - 4) <= 1 and abs(a.tile[1] - 18) <= 1


def test_state_arrays(cfg):
    per = Perceiver(cfg)
    g = Grid(cfg, W, H)
    px, py = g.tile_center(9, 20)
    st = per.state(fake_frame(cfg, 4), [Detection((px - 10, py - 30, px + 10, py - 1), 0.9, "knight", SIDE_ALLY)])
    arr = st.to_arrays(["archer", "knight"])
    assert arr["cls_map"][20, 9] == 2 and arr["side_map"][20, 9] == 1
    assert arr["scalars"][0] == pytest.approx(0.4, abs=0.03)


def test_renderer_layouts(cfg):
    from clashcv.perception.sar import Action
    from clashcv.visualize import Renderer

    per = Perceiver(cfg)
    st = per.state(fake_frame(cfg, 6), [Detection((100, 600, 160, 700), 0.9, "knight", SIDE_ENEMY)])
    act = Action(0, "knight", 0, (4, 20))
    over = Renderer(cfg, height=960).render(fake_frame(cfg, 6), st, [act], 0.5)
    assert over.shape == (960, 432, 3)
    panels = Renderer(cfg, height=960, layout="panels").render(fake_frame(cfg, 6), st, [act], 0.5)
    assert panels.shape[0] == 960 and panels.shape[1] == 432 * 2 + 18 * (960 // 32)
