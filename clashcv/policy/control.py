"""Phone control via adb: tap a hand card, then tap an arena tile."""
from __future__ import annotations

import subprocess
import time

from ..config import ScreenConfig
from ..regions import Grid, to_pixels


class AdbController:
    def __init__(self, cfg: ScreenConfig, screen_w: int, screen_h: int, serial: str | None = None,
                 dry_run: bool = False):
        self.cfg, self.w, self.h = cfg, screen_w, screen_h
        self.grid = Grid(cfg, screen_w, screen_h)
        self.base = ["adb"] + (["-s", serial] if serial else [])
        self.dry_run = dry_run
        self.log: list[tuple[int, int]] = []

    def tap(self, x: int, y: int) -> None:
        self.log.append((x, y))
        if not self.dry_run:
            subprocess.run(self.base + ["shell", "input", "tap", str(x), str(y)], check=True)

    def play(self, slot: int, tile: tuple[int, int], delay: float = 0.08) -> None:
        x0, y0, x1, y1 = to_pixels(self.cfg.cards[slot], self.w, self.h)
        self.tap((x0 + x1) // 2, (y0 + y1) // 2)
        time.sleep(0 if self.dry_run else delay)
        self.tap(*self.grid.tile_center(*tile))
