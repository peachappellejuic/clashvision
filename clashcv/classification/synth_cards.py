"""Synthetic hand-card crops from official card art.

Real hand crops only exist for the cards you've recorded. To recognise the
whole card pool, composite each card's art (``assets/card_art/<card>.png``,
RGBA, e.g. from RoyaleAPI's ``cards-150`` CDN) the way it appears in the hand
slot: blue hand background, slight crop/offset, elixir drop with the cost,
greyed-out "can't afford" state, the small next-card slot, blur, JPEG noise.

Output is an ImageFolder (``<out>/<card>/*.jpg``) that ``train-cls`` consumes.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

OUT_W, OUT_H = 120, 160
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _hand_background(rng, w, h) -> np.ndarray:
    """Blue hand-bar background (BGR) with a vertical gradient and noise."""
    base = np.array([rng.uniform(170, 235), rng.uniform(95, 150), rng.uniform(20, 70)], np.float32)
    grad = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(1.0, 1.15), h, dtype=np.float32)[:, None, None]
    bg = base[None, None, :] * grad * np.ones((h, w, 1), np.float32)
    bg += rng.normal(0, 4, bg.shape).astype(np.float32)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _paste_rgba(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[..., 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (s[..., :3] * a + roi * (1 - a)).astype(np.uint8)


def _elixir_drop(img: np.ndarray, cost: int | None, rng) -> None:
    h, w = img.shape[:2]
    r = int(w * rng.uniform(0.11, 0.15))
    cx, cy = w // 2 + int(rng.integers(-3, 4)), int(h * rng.uniform(0.86, 0.93))
    cv2.circle(img, (cx, cy), r + 2, (60, 20, 60), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (200, 40, 210), -1, cv2.LINE_AA)
    tip = np.array([[cx - r // 2, cy - r + 2], [cx + r // 2, cy - r + 2], [cx + r // 3, cy - int(r * 1.6)]])
    cv2.fillConvexPoly(img, tip, (200, 40, 210), cv2.LINE_AA)
    if cost is not None:
        txt = str(cost)
        fs = r / 14.0
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        org = (cx - tw // 2, cy + th // 2)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_DUPLEX, fs, (30, 30, 30), 4, cv2.LINE_AA)
        cv2.putText(img, txt, org, cv2.FONT_HERSHEY_DUPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)


def render_card(art: np.ndarray, cost: int | None, rng) -> np.ndarray:
    """One synthetic hand-slot crop (BGR, OUT_W x OUT_H) from RGBA card art."""
    canvas = _hand_background(rng, OUT_W, OUT_H)
    # Hand crops are tight: the card fills almost the whole crop and is often clipped.
    scale = rng.uniform(0.85, 1.12) * OUT_H / art.shape[0]
    a = cv2.resize(art, None, fx=scale * rng.uniform(0.93, 1.07), fy=scale, interpolation=cv2.INTER_AREA)
    x = (OUT_W - a.shape[1]) // 2 + int(rng.integers(-10, 11))
    y = (OUT_H - a.shape[0]) // 2 + int(rng.integers(-10, 9))
    if rng.random() < 0.25:  # selection / "evolution ready" glow
        glow = np.zeros_like(a)
        glow[..., :3] = (255, 255, rng.uniform(150, 255))
        glow[..., 3] = cv2.GaussianBlur(cv2.dilate(a[..., 3], np.ones((7, 7), np.uint8)), (0, 0), 3)
        _paste_rgba(canvas, glow, x, y)
    _paste_rgba(canvas, a, x, y)
    if rng.random() < 0.85:
        _elixir_drop(canvas, cost, rng)

    img = canvas
    if rng.random() < 0.35:  # greyed out (not enough elixir); sometimes washed-out white
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = g * rng.uniform(0.5, 0.9) + (rng.uniform(60, 120) if rng.random() < 0.3 else 0)
        img = cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-4, 4)) % 180
    hsv[..., 1] *= rng.uniform(0.8, 1.2)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    if rng.random() < 0.3:  # small rotation / perspective wobble
        M = cv2.getRotationMatrix2D((OUT_W / 2, OUT_H / 2), rng.uniform(-4, 4), rng.uniform(0.95, 1.05))
        img = cv2.warpAffine(img, M, (OUT_W, OUT_H), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.4:  # low-res next-card slot / screen recording
        s = rng.uniform(0.35, 0.7)
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    q = int(rng.integers(35, 95))
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _load_rgba(p: Path) -> np.ndarray | None:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def generate(art_dir: str | Path, out: str | Path, per_class: int = 60,
             real_dir: str | Path | None = None, costs: dict[str, int] | None = None,
             seed: int = 0) -> list[str]:
    """Write ``per_class`` synthetic crops per card art into ``out/<card>/``.

    ``real_dir`` (an ImageFolder of real crops, e.g. ``data/cards``) is copied
    in too, so its classes - including ``empty`` - train alongside the synthetic
    ones. Returns the class list.
    """
    from ..constants import ELIXIR_COST

    costs = {**ELIXIR_COST, **(costs or {})}
    rng = np.random.default_rng(seed)
    out = Path(out)
    arts = sorted(p for p in Path(art_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)
    if not arts:
        raise FileNotFoundError(f"no card art in {art_dir}")
    for p in arts:
        art = _load_rgba(p)
        if art is None:
            continue
        d = out / p.stem
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            cv2.imwrite(str(d / f"synth_{i:04d}.jpg"), render_card(art, costs.get(p.stem), rng))
    if real_dir:
        for d in Path(real_dir).iterdir():
            if not d.is_dir() or d.name.startswith(("_", ".")):
                continue
            (out / d.name).mkdir(parents=True, exist_ok=True)
            for f in d.iterdir():
                if f.suffix.lower() in _IMG_EXTS:
                    shutil.copy2(f, out / d.name / f"real_{f.name}")
    return sorted(d.name for d in out.iterdir() if d.is_dir())
