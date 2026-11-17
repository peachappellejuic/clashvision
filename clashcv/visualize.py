"""Overlays in the style of KataCR.

* Detection boxes in a per-class colour, labelled `<class><side> <track-id> <conf>`
  (side 0 = ally, 1 = enemy, KataCR's convention).
* Hand cards: each slot boxed and labelled with name, elixir cost and confidence.
  Green if affordable, red if not. The next card is shown too.
* HUD: Reward / Frame / Time on top, elixir and the hand list at the bottom.
* `layout="panels"` gives KataCR's eval view side by side:
  [raw screen | annotated screen | 18x32 state grid].
"""
from __future__ import annotations

import colorsys
import zlib
from collections import deque

import cv2
import numpy as np

from .config import ScreenConfig
from .constants import ELIXIR_COST, EMPTY_CARD, SIDE_ALLY, SIDE_ENEMY, TOWER_CLASSES
from .regions import Grid, to_pixels

FONT = cv2.FONT_HERSHEY_SIMPLEX
_SIDE_COLOR = {SIDE_ALLY: (230, 120, 30), SIDE_ENEMY: (40, 40, 220)}  # BGR blue / red


def class_color(name: str) -> tuple[int, int, int]:
    """Stable, distinct BGR colour per class name."""
    h = (zlib.crc32(name.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def label(img, text, x, y, color, scale=0.42, thick=1, above=True) -> None:
    """Text on a filled colour tag, like YOLO/KataCR labels."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x = int(np.clip(x, 0, max(0, img.shape[1] - tw - 2)))
    y0 = int(y - th - base - 2) if above else int(y)
    y0 = int(np.clip(y0, 0, max(0, img.shape[0] - th - base - 2)))
    cv2.rectangle(img, (x, y0), (x + tw + 2, y0 + th + base + 2), color, -1)
    lum = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    cv2.putText(img, text, (x + 1, y0 + th + 1), FONT, scale, (0, 0, 0) if lum > 150 else (255, 255, 255),
                thick, cv2.LINE_AA)


def outlined(img, text, org, scale=0.6, color=(255, 255, 255), thick=1) -> None:
    # Outline via offset copies: in OpenCV 5 a thicker stroke also widens the glyph
    # advance, so a thick black pass would stick out past the text.
    x, y = org
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 2), (2, 0), (0, -2), (-2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_regions(frame: np.ndarray, cfg: ScreenConfig, grid: bool = True) -> np.ndarray:
    """Calibration view: every configured screen region and the tile grid."""
    out = frame.copy()
    h, w = out.shape[:2]

    def box(b, color, text, t=2):
        x0, y0, x1, y1 = to_pixels(b, w, h)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, t)
        cv2.putText(out, text, (x0 + 3, y0 + 18), FONT, 0.55, color, 2)

    if grid:
        g = Grid(cfg, w, h)
        for c in range(g.cols + 1):
            x = int(g.x0 + c * g.tw)
            cv2.line(out, (x, g.y0), (x, g.y1), (200, 200, 200), 1)
        for r in range(g.rows + 1):
            y = int(g.y0 + r * g.th)
            cv2.line(out, (g.x0, y), (g.x1, y), (200, 200, 200), 1)
    box(cfg.timer, (0, 255, 255), "timer")
    box(cfg.arena, (0, 255, 0), "arena")
    box(cfg.hand, (255, 0, 255), "hand")
    for i, b in enumerate(cfg.cards):
        box(b, (255, 255, 0), f"card{i}")
    box(cfg.next_card, (255, 255, 0), "next")
    box(cfg.elixir_bar, (180, 0, 255), "elixir")
    for name, b in cfg.tower_bars.items():
        box(b, (0, 128, 255), name, 1)
    return out


class Renderer:
    """Stateful renderer: remembers recent plays so they stay visible for a while."""

    def __init__(self, cfg: ScreenConfig, height: int = 960, layout: str = "overlay",
                 action_frames: int = 45):
        if layout not in ("overlay", "panels"):
            raise ValueError("layout must be 'overlay' or 'panels'")
        self.cfg, self.height, self.layout = cfg, height, layout
        self.recent: deque = deque()
        self.action_frames = action_frames

    # -- pieces ---------------------------------------------------------------
    def _units(self, img, state, s):
        for u in sorted(state.units, key=lambda u: u.xyxy[3]):
            x0, y0, x1, y1 = (int(v * s) for v in u.xyxy)
            col = class_color(u.cls)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2 if u.cls in TOWER_CLASSES else 1)
            side = "" if u.side not in (SIDE_ALLY, SIDE_ENEMY) else str(u.side)
            tid = f" {u.track_id}" if u.track_id >= 0 else ""
            label(img, f"{u.cls}{side}{tid} {u.conf:.2f}", x0, y0, col)

    def _cards(self, img, state):
        h, w = img.shape[:2]
        for i, (box, name, conf) in enumerate(zip(self.cfg.cards, state.hand, state.hand_conf)):
            x0, y0, x1, y1 = to_pixels(box, w, h)
            cost = ELIXIR_COST.get(name)
            if name == EMPTY_CARD:
                col, sub = (150, 150, 150), f"{conf:.2f}"
            else:
                ok = cost is not None and cost <= state.elixir
                col = (60, 200, 60) if ok else (60, 60, 230)
                sub = f"{cost if cost is not None else '?'}e {conf:.2f}"
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
            # Slots are narrow: name on top (scaled to fit), cost + confidence below.
            scale = min(0.4, 0.4 * (x1 - x0 - 4) / max(1, cv2.getTextSize(name, FONT, 0.4, 1)[0][0]))
            label(img, name, x0, y0, col, scale=max(scale, 0.26))
            label(img, sub, x0, y1, col, scale=0.32, above=False)
        if state.next_card:
            x0, y0, x1, y1 = to_pixels(self.cfg.next_card, w, h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), 2)
            label(img, "next", x0, y0, (0, 200, 255), scale=0.32)
            label(img, state.next_card, x0, y1, (0, 200, 255), scale=0.3, above=False)

    def _hud(self, img, state, reward):
        t = "?" if state.time_left is None else f"{state.time_left // 60}:{state.time_left % 60:02d}"
        outlined(img, f"Reward: {reward:+.4f}", (6, 22), 0.6)
        outlined(img, f"Frame: {state.frame_idx}", (6, 46), 0.6)
        outlined(img, f"Time: {t}", (img.shape[1] - 120, 22), 0.6)
        outlined(img, f"Elixir: {state.elixir:.1f}", (img.shape[1] - 120, 46), 0.6, (230, 120, 255))
        for k, (a, _) in enumerate(self.recent):
            where = "?" if a.tile is None else f"({a.tile[0]},{a.tile[1]})"
            outlined(img, f"PLAY {a.card} @ {where}", (6, 72 + 22 * k), 0.5, (0, 255, 255))

    def _grid_panel(self, state, height):
        cols, rows = self.cfg.grid_size
        cell = height // rows
        panel = np.zeros((height, cols * cell, 3), np.uint8)
        for c in range(cols + 1):
            cv2.line(panel, (c * cell, 0), (c * cell, rows * cell), (90, 90, 90), 1)
        for r in range(rows + 1):
            cv2.line(panel, (0, r * cell), (cols * cell, r * cell), (90, 90, 90), 1)
        cv2.line(panel, (0, rows * cell // 2), (cols * cell, rows * cell // 2), (200, 160, 40), 2)  # river
        for u in state.units:
            c, r = int(np.clip(u.tile[0], 0, cols - 1)), int(np.clip(u.tile[1], 0, rows - 1))
            col = class_color(u.cls)
            cv2.rectangle(panel, (c * cell + 1, r * cell + 1), ((c + 1) * cell - 1, (r + 1) * cell - 1), col, -1)
            if u.side in (SIDE_ALLY, SIDE_ENEMY):
                cv2.putText(panel, str(u.side), (c * cell + 3, r * cell + cell - 6), FONT, 0.4,
                            (255, 255, 255), 1, cv2.LINE_AA)
        for a, _ in self.recent:
            if a.tile is not None:
                c, r = a.tile
                cv2.rectangle(panel, (c * cell, r * cell), ((c + 1) * cell, (r + 1) * cell), (0, 255, 255), 2)
                label(panel, a.card, c * cell, r * cell, (0, 255, 255), 0.38)
        outlined(panel, "state grid 18x32", (6, height - 8), 0.45)
        return panel

    # -- main -----------------------------------------------------------------
    def render(self, frame: np.ndarray, state, actions=(), reward: float = 0.0) -> np.ndarray:
        for a in actions:
            self.recent.append((a, state.frame_idx))
        while self.recent and state.frame_idx - self.recent[0][1] > self.action_frames:
            self.recent.popleft()
        while len(self.recent) > 4:
            self.recent.popleft()

        s = self.height / frame.shape[0]
        base = cv2.resize(frame, (int(round(frame.shape[1] * s)), self.height), interpolation=cv2.INTER_AREA)
        ann = base.copy()
        self._units(ann, state, s)
        self._cards(ann, state)
        self._hud(ann, state, reward)
        if self.layout == "overlay":
            return ann
        return np.hstack([base, ann, self._grid_panel(state, self.height)])


def draw_state(frame, state, actions=(), reward=0.0, cfg: ScreenConfig | None = None,
               layout: str = "overlay", height: int | None = None) -> np.ndarray:
    """One-off render (no action persistence)."""
    return Renderer(cfg or ScreenConfig(), height or frame.shape[0], layout).render(frame, state, actions, reward)
