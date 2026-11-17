"""Pixel-based readers for UI gauges: elixir bar, tower HP bars, timer OCR.

These avoid a neural net where the UI is a fixed-position coloured bar, which is
both faster and more reliable than OCR for elixir and HP fractions.
"""
from __future__ import annotations

import re

import cv2
import numpy as np

from ..config import ScreenConfig
from ..regions import crop

# Elixir fill is a saturated magenta/purple; empty bar is dark.
_ELIXIR = ((135, 90, 110), (170, 255, 255))
# HP bars: ally blue, enemy red; we accept either.
_HP_BLUE = ((95, 110, 110), (125, 255, 255))
_HP_RED1, _HP_RED2 = ((0, 120, 110), (8, 255, 255)), ((170, 120, 110), (180, 255, 255))


def fill_fraction(patch: np.ndarray, masks: list[tuple], col_thresh: float = 0.35) -> float | None:
    """Fraction of columns (left to right) that are 'filled' with the given colours.

    Uses the rightmost filled column rather than a pixel count, so the numeric
    label printed on top of the bar doesn't bias the result.
    Returns None if nothing of that colour is visible (bar hidden).
    """
    if patch.size == 0:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in masks:
        m |= cv2.inRange(hsv, lo, hi)
    cols = (m > 0).mean(axis=0) >= col_thresh
    if not cols.any():
        return None
    # Fill = longest run starting from the left edge, tolerating small gaps (text).
    filled = np.flatnonzero(cols)
    gaps = np.flatnonzero(np.diff(filled) > max(3, len(cols) // 15))
    last = filled[gaps[0]] if len(gaps) else filled[-1]
    if filled[0] > len(cols) * 0.15:
        return None  # not anchored at the left: probably not the bar
    return float((last + 1) / len(cols))


def read_elixir(frame: np.ndarray, cfg: ScreenConfig) -> float:
    """Elixir in [0, 10] (continuous; round() for the displayed integer)."""
    f = fill_fraction(crop(frame, cfg.elixir_bar), [_ELIXIR])
    return round(10.0 * (f or 0.0), 2)


def read_tower_hp(frame: np.ndarray, cfg: ScreenConfig) -> dict[str, float | None]:
    """HP fraction per tower, None when its bar isn't visible (full HP or destroyed)."""
    return {name: fill_fraction(crop(frame, box), [_HP_BLUE, _HP_RED1, _HP_RED2])
            for name, box in cfg.tower_bars.items()}


class TimerOCR:
    """Reads the 'm:ss' clock. Tries PaddleOCR, EasyOCR, then pytesseract."""

    def __init__(self):
        self.backend, self._ocr = None, None
        for name in ("paddleocr", "easyocr", "pytesseract"):
            try:
                mod = __import__(name)
            except ImportError:
                continue
            if name == "paddleocr":
                self._ocr = mod.PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
            elif name == "easyocr":
                self._ocr = mod.Reader(["en"], gpu=False, verbose=False)
            else:
                self._ocr = mod
            self.backend = name
            break

    def text(self, img: np.ndarray) -> str:
        if self.backend == "paddleocr":
            res = self._ocr.ocr(img, cls=False) or []
            return " ".join(t[1][0] for line in res if line for t in line)
        if self.backend == "easyocr":
            return " ".join(self._ocr.readtext(img, detail=0, allowlist="0123456789:"))
        if self.backend == "pytesseract":
            g = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return self._ocr.image_to_string(g, config="--psm 7 -c tessedit_char_whitelist=0123456789:")
        return ""

    def seconds_left(self, frame: np.ndarray, cfg: ScreenConfig) -> int | None:
        if self.backend is None:
            return None
        return parse_clock(self.text(crop(frame, cfg.timer)))


def parse_clock(text: str) -> int | None:
    m = re.search(r"(\d)\s*[:.]\s*(\d{2})", text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"\b(\d{1,3})\b", text)  # overtime / last seconds show bare numbers
    return int(m.group(1)) if m else None
