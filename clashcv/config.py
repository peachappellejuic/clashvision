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


def _cards(y0: float, y1: float) -> list[Box]:
    # Four hand-card slots across the bottom bar (measured from KataCR footage).
    xs = [(0.222, 0.385), (0.416, 0.579), (0.610, 0.773), (0.804, 0.967)]
    return [(a, y0, b, y1) for a, b in xs]


def _default_cards() -> list[Box]:
    return _cards(0.868, 0.963)


# Region presets per screen aspect ratio (height / width). Arena + timer boxes
# are KataCR's `split_bbox_params`; the bottom UI is anchored to the screen
# bottom, so it barely moves between ratios.
ASPECT_PRESETS: dict[float, dict] = {
    2.13: dict(timer=(0.845, 0.037, 1.0, 0.075), arena=(0.026, 0.048, 0.986, 0.758),
               hand=(0.0, 0.845, 1.0, 1.0), cards=_cards(0.862, 0.962)),
    2.16: dict(timer=(0.835, 0.063, 1.0, 0.101), arena=(0.021, 0.073, 0.981, 0.773),
               hand=(0.0, 0.808, 1.0, 1.0), cards=_cards(0.865, 0.962)),
    2.22: dict(timer=(0.835, 0.058, 1.0, 0.096), arena=(0.020, 0.070, 0.980, 0.760),
               hand=(0.0, 0.850, 1.0, 1.0), cards=_cards(0.868, 0.963)),
}


@dataclass
class ScreenConfig:
    # Coarse parts (like KataCR's part1/part2/part3 split).
    timer: Box = (0.835, 0.058, 1.0, 0.096)
    arena: Box = (0.020, 0.070, 0.980, 0.760)
    hand: Box = (0.0, 0.850, 1.0, 1.0)

    # Fine regions inside the hand bar.
    cards: list[Box] = field(default_factory=_default_cards)
    next_card: Box = (0.050, 0.945, 0.135, 0.992)
    elixir_bar: Box = (0.278, 0.970, 0.976, 0.992)

    # The 18x32 tile grid, relative to the full frame (KataCR maps it onto the arena crop).
    grid_box: Box = (0.020, 0.070, 0.980, 0.760)
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

    def apply_aspect(self, frame_w: int, frame_h: int) -> float:
        """Switch to the closest aspect-ratio preset (in place). Returns the preset used."""
        ratio = frame_h / frame_w
        key = min(ASPECT_PRESETS, key=lambda k: abs(k - ratio))
        for k, v in ASPECT_PRESETS[key].items():
            setattr(self, k, list(v) if k == "cards" else v)
        self.grid_box = self.arena
        return key

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
