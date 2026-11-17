"""Debug overlays: regions, tile grid, detections, and a state HUD."""
from __future__ import annotations

import cv2
import numpy as np

from .config import ScreenConfig
from .constants import SIDE_ALLY, SIDE_ENEMY
from .regions import Grid, to_pixels

_SIDE_COLOR = {SIDE_ALLY: (255, 140, 0), SIDE_ENEMY: (40, 40, 230)}


def draw_regions(frame: np.ndarray, cfg: ScreenConfig, grid: bool = True) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    def box(b, color, label, t=2):
        x0, y0, x1, y1 = to_pixels(b, w, h)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, t)
        cv2.putText(out, label, (x0 + 3, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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


def draw_state(frame: np.ndarray, state, actions=(), reward: float = 0.0) -> np.ndarray:
    out = frame.copy()
    for u in state.units:
        x0, y0, x1, y1 = map(int, u.xyxy)
        col = _SIDE_COLOR.get(u.side, (0, 255, 0))
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        cv2.putText(out, f"{u.cls}#{u.track_id} {u.conf:.2f}", (x0, max(12, y0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    lines = [f"t={state.time_left}  elixir={state.elixir:.1f}  r={reward:+.2f}",
             "hand: " + ", ".join(state.hand)]
    lines += [f"PLAY {a.card} slot{a.slot} @ {a.tile}" for a in actions]
    for i, s in enumerate(lines):
        cv2.putText(out, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, s, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out
