"""Build RGBA sprite cut-outs from real frames, to feed the generator.

Given frames plus YOLO-format labels (e.g. a few hundred hand-labelled frames
from Label Studio / CVAT / Roboflow), crop each box and separate foreground
with GrabCut, producing `segments/<class>/<n>.png` with an alpha channel.
Review the results and delete bad cut-outs - quality here drives detector
quality.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def grabcut_rgba(crop: np.ndarray, iters: int = 4, margin: float = 0.06) -> np.ndarray | None:
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    mask = np.zeros((h, w), np.uint8)
    mx, my = max(1, int(w * margin)), max(1, int(h * margin))
    rect = (mx, my, w - 2 * mx, h - 2 * my)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(crop, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if fg.mean() < 255 * 0.08:
        return None  # segmentation failed
    return np.dstack([crop, fg])


def extract_segments(images_dir: str | Path, labels_dir: str | Path, names: list[str],
                     out_dir: str | Path, pad: float = 0.08) -> int:
    images_dir, labels_dir, out_dir = Path(images_dir), Path(labels_dir), Path(out_dir)
    count = 0
    for img_path in sorted(images_dir.iterdir()):
        lab = labels_dir / f"{img_path.stem}.txt"
        if not lab.exists() or (img := cv2.imread(str(img_path))) is None:
            continue
        H, W = img.shape[:2]
        for j, line in enumerate(lab.read_text().split("\n")):
            parts = line.split()
            if len(parts) != 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:])
            bw, bh = bw * (1 + pad), bh * (1 + pad)
            x0, y0 = int(max(0, (cx - bw / 2) * W)), int(max(0, (cy - bh / 2) * H))
            x1, y1 = int(min(W, (cx + bw / 2) * W)), int(min(H, (cy + bh / 2) * H))
            rgba = grabcut_rgba(img[y0:y1, x0:x1])
            if rgba is None:
                continue
            d = out_dir / names[c]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{img_path.stem}_{j}.png"), rgba)
            count += 1
    return count
