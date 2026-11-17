"""Ally / enemy assignment.

KataCR adds a "belong" output to YOLO. Without retraining the head, a robust
heuristic works well: every unit has a level badge / HP bar drawn above it,
blue for the player and red for the opponent. We look at a strip just above the
box and compare blue vs red pixel counts, falling back to the unit's position.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..constants import SIDE_ALLY, SIDE_ENEMY
from .types import Detection

# HSV ranges (OpenCV: H in [0,180]).
_BLUE = ((95, 120, 120), (125, 255, 255))
_RED1 = ((0, 140, 120), (8, 255, 255))
_RED2 = ((170, 140, 120), (180, 255, 255))


def color_votes(patch: np.ndarray) -> tuple[int, int]:
    if patch.size == 0:
        return 0, 0
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, *_BLUE)
    red = cv2.inRange(hsv, *_RED1) | cv2.inRange(hsv, *_RED2)
    return int(np.count_nonzero(blue)), int(np.count_nonzero(red))


def infer_side(frame: np.ndarray, det: Detection, frame_h: int | None = None,
               min_pixels: int = 6) -> int:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = det.xyxy
    bh = max(y1 - y0, 1.0)
    # Strip covering the top of the box and a bit above it (where the bar sits).
    sy0, sy1 = int(max(0, y0 - 0.35 * bh)), int(min(h, y0 + 0.25 * bh))
    sx0, sx1 = int(max(0, x0)), int(min(w, x1))
    blue, red = color_votes(frame[sy0:sy1, sx0:sx1])
    if max(blue, red) >= min_pixels and blue != red:
        return SIDE_ALLY if blue > red else SIDE_ENEMY
    # Fallback: own half of the arena is the bottom half.
    return SIDE_ALLY if det.foot[1] > (frame_h or h) * 0.44 else SIDE_ENEMY
