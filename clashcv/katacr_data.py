"""Build a training set from KataCR's public dataset (MIT):
https://github.com/wty-yy/Clash-Royale-Detection-Dataset

    git clone --depth 1 https://github.com/wty-yy/Clash-Royale-Detection-Dataset data/katacr-dataset
    python -m clashcv prepare-katacr data/katacr-dataset --out data/cr --n-synth 4000

Produces a YOLO dataset of
  * ~7k real hand-labelled arena frames (images/part2, Labelme JSON), and
  * synthetic frames composited from its 4.6k sprites (images/segment),
with classes named KataCR-style: `<unit><side>` (`knight0` ally, `knight1`
enemy), plus unsided UI classes (`clock`, `text`, `elixir`, ...).
Also exposes the card crops (images/card_classification) for `train-cls`.
"""
from __future__ import annotations

import json
import os
import zlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .constants import KATACR_CLASSES, split_side

SKIP_SEGMENT_DIRS = {"backgrounds", "background-items"}


def parse_label(label: str) -> str | None:
    """'king-tower1_freeze' -> 'king-tower1'; 'clock' -> 'clock'; unknown -> None."""
    head = label.split("_")[0]
    base, side = split_side(head)
    if base not in KATACR_CLASSES:
        return None
    return head if side >= 0 else base


def class_list(names: set[str]) -> list[str]:
    """Stable order: KataCR index order, then ally/enemy/unsided."""
    order = []
    for c in KATACR_CLASSES:
        for n in (f"{c}0", f"{c}1", c):
            if n in names:
                order.append(n)
    return order


def _real_items(root: Path) -> tuple[list[tuple[Path, list[tuple[str, tuple]]]], Counter]:
    items, skipped = [], Counter()
    for j in sorted(root.glob("images/part2/**/*.json")):
        d = json.loads(j.read_text())
        img = j.with_name(d.get("imagePath") or f"{j.stem}.jpg")
        if not img.exists():
            img = j.with_suffix(".jpg")
            if not img.exists():
                continue
        W, H = d["imageWidth"], d["imageHeight"]
        boxes = []
        for s in d["shapes"]:
            if s.get("shape_type") != "rectangle" or len(s["points"]) != 2:
                continue
            name = parse_label(s["label"])
            if name is None:
                skipped[s["label"]] += 1
                continue
            (x0, y0), (x1, y1) = s["points"]
            x0, x1 = sorted((min(max(x0, 0), W), min(max(x1, 0), W)))
            y0, y1 = sorted((min(max(y0, 0), H), min(max(y1, 0), H)))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append((name, ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)))
        items.append((img, boxes))
    return items, skipped


def load_katacr_segments(root: Path) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """Sprites keyed by sided class name, plus unlabelled decoration sprites."""
    segs: dict[str, list[np.ndarray]] = {}
    distractors = []
    for p in sorted(root.glob("images/segment/*/*.png")):
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if im is None or im.ndim != 3 or im.shape[2] != 4:
            continue
        if p.parent.name == "background-items":
            distractors.append(im)
            continue
        if p.parent.name in SKIP_SEGMENT_DIRS:
            continue
        parts = p.stem.split("_")
        if len(parts) < 2 or parts[0] not in KATACR_CLASSES or parts[1] not in ("0", "1"):
            continue
        segs.setdefault(f"{parts[0]}{parts[1]}", []).append(im)
    return segs, distractors


def prepare(dataset: str | Path, out: str | Path, n_synth: int = 4000, n_synth_val: int = 200,
            val_mod: int = 10, seed: int = 0) -> Path:
    from .generation import SyntheticGenerator, load_backgrounds

    root, out = Path(dataset), Path(out)
    if not (root / "images" / "part2").exists():
        raise FileNotFoundError(f"{root} doesn't look like the KataCR dataset (no images/part2)")
    real, skipped = _real_items(root)
    segs, distractors = load_katacr_segments(root)
    names = class_list({n for _, bx in real for n, _ in bx} | set(segs))
    ids = {n: i for i, n in enumerate(names)}
    print(f"real frames: {len(real)}  sprites: {sum(map(len, segs.values()))} in {len(segs)} classes  "
          f"decorations: {len(distractors)}  -> {len(names)} detector classes")
    if skipped:
        print(f"skipped unknown labels: {dict(skipped.most_common(8))}")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Real frames: hold out whole recording folders for validation (no near-duplicate leakage).
    counts = Counter()
    for img, boxes in real:
        rel = img.relative_to(root / "images" / "part2")
        split = "val" if zlib.crc32(str(rel.parent).encode()) % val_mod == 0 else "train"
        stem = "real_" + str(rel.with_suffix("")).replace(os.sep, "__")
        link = out / "images" / split / f"{stem}{img.suffix}"
        if not link.exists():
            link.symlink_to(img.resolve())
        (out / "labels" / split / f"{stem}.txt").write_text(
            "".join(f"{ids[n]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for n, (x, y, w, h) in boxes))
        counts[split] += 1
    print(f"real: {counts['train']} train / {counts['val']} val")

    if n_synth or n_synth_val:
        gen = SyntheticGenerator(segs, load_backgrounds(root / "images" / "segment" / "backgrounds"),
                                 names, size=(568, 896), units_range=(5, 30), seed=seed,
                                 distractors=distractors)
        for split, n in (("train", n_synth), ("val", n_synth_val)):
            for i in range(n):
                img, labels = gen.sample()
                stem = f"synth_{i:06d}"
                cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
                if (i + 1) % 500 == 0:
                    print(f"synthetic {split}: {i + 1}/{n}")

    yaml_path = out / "data.yaml"
    yaml_path.write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
                         + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    (out / "classes.json").write_text(json.dumps(names))
    print(f"wrote {yaml_path}")
    return yaml_path
