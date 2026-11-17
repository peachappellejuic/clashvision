import cv2
import numpy as np
import pytest

from clashcv.config import ScreenConfig
from clashcv.regions import to_pixels

W, H = 1080, 2400
ELIXIR_BGR = (220, 40, 200)  # magenta-ish, inside the elixir HSV range


def fake_frame(cfg: ScreenConfig, elixir: float = 5.0, cards=None) -> np.ndarray:
    img = np.full((H, W, 3), (60, 110, 70), np.uint8)  # greenish arena
    x0, y0, x1, y1 = to_pixels(cfg.elixir_bar, W, H)
    img[y0:y1, x0:x1] = (30, 20, 30)
    fill = int((x1 - x0) * elixir / 10)
    img[y0:y1, x0:x0 + fill] = ELIXIR_BGR
    for b, card in zip(cfg.cards, cards or []):
        cx0, cy0, cx1, cy1 = to_pixels(b, W, H)
        img[cy0:cy1, cx0:cx1] = card_image(card, (cx1 - cx0, cy1 - cy0))
    return img


def card_image(name: str, size) -> np.ndarray:
    """Deterministic distinct pattern per card name."""
    w, h = size
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        p2 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        cv2.rectangle(img, p1, p2, c, -1)
    return img


def sprite(color, size=(40, 60)) -> np.ndarray:
    w, h = size
    s = np.zeros((h, w, 4), np.uint8)
    cv2.ellipse(s, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, (*color, 255), -1)
    return s


@pytest.fixture
def cfg():
    return ScreenConfig()
