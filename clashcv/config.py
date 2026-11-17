"""Screen geometry configuration.

All regions are *relative* boxes (x0, y0, x1, y1) in [0, 1] of the full portrait
phone frame, so the same config works for any resolution with the same aspect
ratio. Defaults are tuned for a 1080x2400 (≈ 9:20) screen, like KataCR's
reference device. Use `python -m clashcv calibrate <frame.png>` to visualise
and tweak them, then save overrides to a YAML file and pass `--config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

Box = tuple[float, float, float, float]


def _default_cards() -> list[Box]:
    # Four hand-card slots across the bottom bar.
    x0, w, gap = 0.215, 0.175, 0.012
    return [(x0 + i * (w + gap), 0.835, x0 + i * (w + gap) + w, 0.945) for i in range(4)]


@dataclass
class ScreenConfig:
    # Coarse parts (like KataCR's part1/part2/part3 split).
    timer: Box = (0.78, 0.065, 0.99, 0.105)
    arena: Box = (0.0, 0.075, 1.0, 0.805)
    hand: Box = (0.0, 0.805, 1.0, 1.0)

    # Fine regions inside the hand bar.
    cards: list[Box] = field(default_factory=_default_cards)
    next_card: Box = (0.04, 0.905, 0.16, 0.985)
    elixir_bar: Box = (0.27, 0.955, 0.985, 0.985)

    # The playable 18x32 tile grid, relative to the full frame.
    grid_box: Box = (0.025, 0.115, 0.975, 0.775)
    grid_size: tuple[int, int] = (18, 32)  # (cols, rows)

    # Tower HP bars: name -> box (ally towers at the bottom).
    tower_bars: dict[str, Box] = field(default_factory=lambda: {
        "enemy_king": (0.42, 0.095, 0.62, 0.110),
        "enemy_left": (0.15, 0.215, 0.32, 0.228),
        "enemy_right": (0.68, 0.215, 0.85, 0.228),
        "ally_left": (0.15, 0.640, 0.32, 0.653),
        "ally_right": (0.68, 0.640, 0.85, 0.653),
        "ally_king": (0.42, 0.765, 0.62, 0.780),
    })

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None) -> "ScreenConfig":
        cfg = cls()
        if path is None:
            return cfg
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown config key: {k}")
            if k == "cards":
                v = [tuple(b) for b in v]
            elif k == "tower_bars":
                v = {n: tuple(b) for n, b in v.items()}
            elif isinstance(v, list):
                v = tuple(v)
            setattr(cfg, k, v)
        return cfg

    def save(self, path: str | Path) -> None:
        import yaml

        def plain(o):
            if isinstance(o, tuple):
                return [plain(x) for x in o]
            if isinstance(o, list):
                return [plain(x) for x in o]
            if isinstance(o, dict):
                return {k: plain(v) for k, v in o.items()}
            return o

        Path(path).write_text(yaml.safe_dump(plain(self.to_dict()), sort_keys=False))
