"""Synthetic detection dataset generator (KataCR `build_dataset/generator.py`).

Inputs
  segments/<class>/*.png   RGBA cut-outs of units/towers/spells (alpha = mask)
  backgrounds/*.png        empty arena crops (same crop as ScreenConfig.arena)

Each image: random background -> towers at their fixed slots -> N random
sprites, drawn back-to-front by foot y (so nearer units occlude farther ones,
like the game renders) -> YOLO labels. Heavily occluded objects are dropped
from the labels so the detector isn't trained on invisible targets.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..constants import SIDE_UNKNOWN, TOWER_CLASSES, split_side

_IMG_EXTS = {".png", ".webp"}

# Relative (x, y) of tower *foot* points inside the arena crop, with the side
# that owns them (1 = enemy at the top, 0 = ally at the bottom).
TOWER_SLOTS = {
    "king": [((0.5, 0.16), 1), ((0.5, 0.95), 0)],
    "queen": [((0.22, 0.27), 1), ((0.78, 0.27), 1), ((0.22, 0.83), 0), ((0.78, 0.83), 0)],
}


@dataclass
class Sprite:
    cls: str
    rgba: np.ndarray  # H x W x 4


@dataclass
class Placed:
    cls_id: int
    img: np.ndarray
    x: int  # top-left in canvas
    y: int

    @property
    def foot_y(self) -> int:
        return self.y + self.img.shape[0]


def load_segments(seg_dir: str | Path) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    for d in sorted(Path(seg_dir).iterdir()):
        if not d.is_dir():
            continue
        imgs = []
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in _IMG_EXTS:
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if im is not None and im.ndim == 3 and im.shape[2] == 4:
                    imgs.append(im)
        if imgs:
            out[d.name] = imgs
    if not out:
        raise FileNotFoundError(f"no RGBA segments found under {seg_dir}/<class>/*.png")
    return out


def load_backgrounds(bg_dir: str | Path) -> list[np.ndarray]:
    bgs = [cv2.imread(str(p)) for p in sorted(Path(bg_dir).iterdir())
           if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    bgs = [b for b in bgs if b is not None]
    if not bgs:
        raise FileNotFoundError(f"no background images in {bg_dir}")
    return bgs


def _augment_sprite(rgba: np.ndarray, rng: random.Random, scale: float) -> np.ndarray:
    s = scale * rng.uniform(0.85, 1.15)
    h, w = rgba.shape[:2]
    rgba = cv2.resize(rgba, (max(2, int(w * s)), max(2, int(h * s))), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    if rng.random() < 0.3:  # brightness jitter on colour channels only
        rgba = rgba.copy()
        rgba[..., :3] = cv2.convertScaleAbs(rgba[..., :3], alpha=rng.uniform(0.8, 1.2),
                                            beta=rng.uniform(-15, 15))
    return np.ascontiguousarray(rgba)


class SyntheticGenerator:
    def __init__(self, segments: dict[str, list[np.ndarray]], backgrounds: list[np.ndarray],
                 class_names: list[str] | None = None, size: tuple[int, int] = (576, 896),
                 sprite_scale: float = 1.0, units_range: tuple[int, int] = (4, 25),
                 min_visible: float = 0.35, seed: int | None = None,
                 distractors: list[np.ndarray] | None = None, distractor_range: tuple[int, int] = (0, 4)):
        """Class names may carry a side digit ("king-tower1"): towers are then
        placed on that side's slots. `distractors` are RGBA decorations pasted
        unlabelled under everything else."""
        self.segments, self.backgrounds = segments, backgrounds
        self.distractors, self.distractor_range = distractors or [], distractor_range
        self.names = class_names or sorted(segments)
        self.ids = {c: i for i, c in enumerate(self.names)}
        self.size, self.sprite_scale = size, sprite_scale
        self.units_range, self.min_visible = units_range, min_visible
        self.rng = random.Random(seed)
        usable = [c for c in segments if c in self.ids]
        self.unit_classes = [c for c in usable if split_side(c)[0] not in TOWER_CLASSES]
        self.towers: dict[tuple[str, int], list[str]] = {}
        for c in usable:
            base, side = split_side(c)
            if base in TOWER_CLASSES:
                kind = "king" if base == "king-tower" else "queen"
                for sd in ((0, 1) if side == SIDE_UNKNOWN else (side,)):
                    self.towers.setdefault((kind, sd), []).append(c)
        if not self.unit_classes:
            raise ValueError("need at least one non-tower segment class")

    def _place(self, cls: str, foot: tuple[float, float]) -> Placed:
        img = _augment_sprite(self.rng.choice(self.segments[cls]), self.rng, self.sprite_scale)
        W, H = self.size
        h, w = img.shape[:2]
        return Placed(self.ids[cls], img, int(foot[0] * W - w / 2), int(foot[1] * H - h))

    def sample(self) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
        """Returns (BGR image, [(cls_id, cx, cy, w, h) normalised])."""
        rng, (W, H) = self.rng, self.size
        bg = cv2.resize(rng.choice(self.backgrounds), (W, H), interpolation=cv2.INTER_AREA)
        canvas = cv2.convertScaleAbs(bg, alpha=rng.uniform(0.9, 1.1), beta=rng.uniform(-10, 10))

        for _ in range(rng.randint(*self.distractor_range) if self.distractors else 0):
            d = _augment_sprite(rng.choice(self.distractors), rng, self.sprite_scale)
            dh, dw = d.shape[:2]
            x, y = int(rng.uniform(0, W) - dw / 2), int(rng.uniform(0, H) - dh / 2)
            x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + dw, W), min(y + dh, H)
            if x1 > x0 and y1 > y0:
                sub = d[y0 - y:y1 - y, x0 - x:x1 - x]
                a = sub[..., 3:4].astype(np.float32) / 255.0
                canvas[y0:y1, x0:x1] = (sub[..., :3] * a + canvas[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)

        objs: list[Placed] = []
        for kind, p_present in (("king", 0.9), ("queen", 0.8)):
            for slot, side in TOWER_SLOTS[kind]:
                choices = self.towers.get((kind, side))
                if choices and rng.random() < p_present:
                    objs.append(self._place(rng.choice(choices), slot))
        for _ in range(rng.randint(*self.units_range)):
            cls = rng.choice(self.unit_classes)  # class-balanced sampling
            objs.append(self._place(cls, (rng.uniform(0.04, 0.96), rng.uniform(0.06, 0.98))))

        objs.sort(key=lambda o: o.foot_y)  # painter's algorithm
        owner = np.full((H, W), -1, np.int32)
        areas = []
        for k, o in enumerate(objs):
            h, w = o.img.shape[:2]
            x0, y0, x1, y1 = max(o.x, 0), max(o.y, 0), min(o.x + w, W), min(o.y + h, H)
            if x1 <= x0 or y1 <= y0:
                areas.append((0, None))
                continue
            sub = o.img[y0 - o.y:y1 - o.y, x0 - o.x:x1 - o.x]
            a = sub[..., 3:4].astype(np.float32) / 255.0
            roi = canvas[y0:y1, x0:x1]
            canvas[y0:y1, x0:x1] = (sub[..., :3] * a + roi * (1 - a)).astype(np.uint8)
            mask = sub[..., 3] > 127
            owner[y0:y1, x0:x1][mask] = k
            total = int(np.count_nonzero(o.img[..., 3] > 127))
            areas.append((total, (x0, y0, x1, y1)))

        visible = np.bincount(owner.ravel() + 1, minlength=len(objs) + 1)[1:]
        labels = []
        for k, (o, (total, box)) in enumerate(zip(objs, areas)):
            if box is None or total == 0 or visible[k] / total < self.min_visible:
                continue
            # Full (amodal) box of the sprite's opaque pixels, clipped to the canvas.
            ay, ax = np.nonzero(o.img[..., 3] > 127)
            bx0, bx1 = np.clip([ax.min() + o.x, ax.max() + o.x + 1], 0, W)
            by0, by1 = np.clip([ay.min() + o.y, ay.max() + o.y + 1], 0, H)
            if bx1 - bx0 < 3 or by1 - by0 < 3:
                continue
            labels.append((o.cls_id, (bx0 + bx1) / 2 / W, (by0 + by1) / 2 / H,
                           (bx1 - bx0) / W, (by1 - by0) / H))
        return canvas, labels

    def write_dataset(self, out_dir: str | Path, n_train: int, n_val: int) -> Path:
        out = Path(out_dir)
        for split, n in (("train", n_train), ("val", n_val)):
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img, labels = self.sample()
                stem = f"{split}_{i:06d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
        yaml_path = out / "data.yaml"
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(self.names))
        yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n")
        (out / "classes.json").write_text(json.dumps(self.names))
        return yaml_path
