"""Self-contained demo: no game assets needed.

`python -m clashcv demo` builds (and caches) everything on first run:
  1. procedural sprites + an empty arena background
  2. a synthetic YOLO dataset from them (the same generator used for real data)
  3. a YOLOv8n fine-tuned on that dataset
  4. card templates and an animated 1080x2400 "gameplay" video
then runs the full perception pipeline on the video with a live window.
"""
from __future__ import annotations

import shutil
import zlib
from pathlib import Path

import cv2
import numpy as np

from .config import ScreenConfig
from .regions import to_pixels

W, H = 1080, 2400
ARENA_SIZE = (576, 896)  # generator canvas (w, h)
ELIXIR_BGR = (220, 40, 200)
DEMO_HANDS = [["hog-rider", "musketeer", "cannon", "ice-golem"],
              ["fireball", "musketeer", "cannon", "ice-golem"]]
SPRITES = {  # class -> (BGR colour, (w, h) at ARENA_SIZE scale)
    "knight": ((200, 50, 50), (40, 60)),
    "archer": ((50, 200, 50), (30, 50)),
    "king-tower": ((120, 120, 120), (120, 120)),
}
BG_BGR = (60, 110, 70)


def sprite(color, size=(40, 60)) -> np.ndarray:
    """RGBA filled ellipse."""
    w, h = size
    s = np.zeros((h, w, 4), np.uint8)
    cv2.ellipse(s, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, (*color, 255), -1)
    return s


def card_image(name: str, size) -> np.ndarray:
    """Deterministic distinct pattern per card name."""
    w, h = size
    # zlib.crc32, not hash(): str hashes are randomised per process.
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(6):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p1 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        p2 = tuple(int(v) for v in (rng.integers(0, w), rng.integers(0, h)))
        cv2.rectangle(img, p1, p2, c, -1)
    return img


def fake_frame(cfg: ScreenConfig, elixir: float = 5.0, cards=None) -> np.ndarray:
    """Blank 1080x2400 phone frame with an elixir bar and hand cards."""
    img = np.full((H, W, 3), BG_BGR, np.uint8)
    x0, y0, x1, y1 = to_pixels(cfg.elixir_bar, W, H)
    img[y0:y1, x0:x1] = (30, 20, 30)
    img[y0:y1, x0:x0 + int((x1 - x0) * elixir / 10)] = ELIXIR_BGR
    for b, card in zip(cfg.cards, cards or []):
        cx0, cy0, cx1, cy1 = to_pixels(b, W, H)
        img[cy0:cy1, cx0:cx1] = card_image(card, (cx1 - cx0, cy1 - cy0))
    return img


def _paste(img, s, cx, foot):
    h, w = s.shape[:2]
    x, y = int(cx - w / 2), int(foot - h)
    a = s[..., 3:4] / 255.0
    img[y:y + h, x:x + w] = (s[..., :3] * a + img[y:y + h, x:x + w] * (1 - a)).astype(np.uint8)


def make_assets(out: Path, cfg: ScreenConfig) -> None:
    for name, (col, sz) in SPRITES.items():
        (out / "segments" / name).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "segments" / name / "0.png"), sprite(col, sz))
    (out / "backgrounds").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "backgrounds" / "0.png"), np.full((ARENA_SIZE[1], ARENA_SIZE[0], 3), BG_BGR, np.uint8))
    (out / "cards").mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = to_pixels(cfg.cards[0], W, H)
    for n in {c for hand in DEMO_HANDS for c in hand}:
        cv2.imwrite(str(out / "cards" / f"{n}.png"), card_image(n, (x1 - x0, y1 - y0)))


def render_video(path: Path, cfg: ScreenConfig, n_frames: int = 450, fps: int = 30) -> None:
    ax0, ay0, ax1, ay1 = to_pixels(cfg.arena, W, H)
    AW, AH = ax1 - ax0, ay1 - ay0
    sx, sy = AW / ARENA_SIZE[0], AH / ARENA_SIZE[1]
    spr = {n: cv2.resize(sprite(c, sz), (int(sz[0] * sx), int(sz[1] * sy))) for n, (c, sz) in SPRITES.items()}
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for t in range(n_frames):
        f = fake_frame(cfg, (t / fps * 0.7) % 10, DEMO_HANDS[(t // 150) % 2])
        ar = f[ay0:ay1, ax0:ax1]
        _paste(ar, spr["king-tower"], AW * 0.5, AH * 0.16)
        _paste(ar, spr["king-tower"], AW * 0.5, AH * 0.95)
        for k in range(3):  # knights march down, archers march up
            p = t / n_frames
            _paste(ar, spr["knight"], AW * (0.2 + 0.3 * k), AH * (0.25 + 0.54 * ((p + k * 0.2) % 1)))
            _paste(ar, spr["archer"], AW * (0.3 + 0.25 * k), AH * (0.9 - 0.6 * ((p + k * 0.3) % 1)))
        vw.write(f)
    vw.release()


def build(out: str | Path = "runs/demo", epochs: int = 20, device: str | None = None,
          rebuild: bool = False) -> dict[str, Path]:
    """Create (or reuse) demo assets, dataset, weights and video. Returns their paths."""
    out = Path(out)
    if rebuild and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = ScreenConfig()
    paths = {"video": out / "demo.mp4", "weights": out / "detector.pt", "cards": out / "cards"}

    if not (out / "segments").exists() or not paths["cards"].exists():
        print("[demo] drawing sprites and card templates")
        make_assets(out, cfg)

    if not paths["weights"].exists():
        from .generation import SyntheticGenerator, load_backgrounds, load_segments

        print("[demo] generating synthetic dataset")
        gen = SyntheticGenerator(load_segments(out / "segments"), load_backgrounds(out / "backgrounds"),
                                 size=ARENA_SIZE, units_range=(3, 10), seed=1)
        data = gen.write_dataset(out / "dataset", 160, 20)

        print(f"[demo] training YOLOv8n for {epochs} epochs (a few minutes on CPU)")
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.train(data=str(data), epochs=epochs, imgsz=448, batch=16, device=device,
                    project=str((out / "train").resolve()), name="det", exist_ok=True,
                    plots=False, verbose=False, hsv_h=0.0, flipud=0.0)
        shutil.copy(Path(model.trainer.save_dir) / "weights" / "best.pt", paths["weights"])

    if not paths["video"].exists():
        print("[demo] rendering demo video")
        render_video(paths["video"], cfg)
    return paths
