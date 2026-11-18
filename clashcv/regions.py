"""Cropping helpers and pixel <-> tile coordinate conversion."""
from __future__ import annotations

import cv2
import numpy as np

from .config import Box, ScreenConfig


def to_pixels(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h))


def crop(frame: np.ndarray, box: Box) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = to_pixels(box, w, h)
    return frame[y0:y1, x0:x1]


def split_parts(frame: np.ndarray, cfg: ScreenConfig) -> dict[str, np.ndarray]:
    """KataCR-style split: timer (part1), arena (part2), hand (part3)."""
    return {"timer": crop(frame, cfg.timer), "arena": crop(frame, cfg.arena), "hand": crop(frame, cfg.hand)}


class Grid:
    """Maps between full-frame pixels and the 18x32 arena tile grid.

    Tile (0, 0) is the top-left (enemy side). x grows right, y grows down.
    Tile coordinates are floats; the integer part is the tile index.
    """

    def __init__(self, cfg: ScreenConfig, frame_w: int, frame_h: int):
        self.cols, self.rows = cfg.grid_size
        self.x0, self.y0, self.x1, self.y1 = to_pixels(cfg.grid_box, frame_w, frame_h)
        self.tw = (self.x1 - self.x0) / self.cols
        self.th = (self.y1 - self.y0) / self.rows

    def pixel_to_tile(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / self.tw, (y - self.y0) / self.th

    def tile_to_pixel(self, tx: float, ty: float) -> tuple[int, int]:
        return int(round(self.x0 + tx * self.tw)), int(round(self.y0 + ty * self.th))

    def tile_center(self, col: int, row: int) -> tuple[int, int]:
        return self.tile_to_pixel(col + 0.5, row + 0.5)

    def clamp_tile(self, tx: float, ty: float) -> tuple[int, int]:
        return int(np.clip(tx, 0, self.cols - 1)), int(np.clip(ty, 0, self.rows - 1))


def find_screen(frame: np.ndarray, thresh: int = 18) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) of the non-black content, e.g. a portrait phone
    recording inside a landscape video with black bars. None if no bars."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cols = np.flatnonzero(g.mean(axis=0) > thresh)
    rows = np.flatnonzero(g.mean(axis=1) > thresh)
    if len(cols) == 0 or len(rows) == 0:
        return None
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    h, w = g.shape
    if (x1 - x0) > 0.97 * w and (y1 - y0) > 0.97 * h:
        return None
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)
