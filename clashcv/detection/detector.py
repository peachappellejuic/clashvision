"""YOLOv8 detectors (Ultralytics) with KataCR-style model combos.

Clash Royale has ~150 object classes, from tiny skeletons to king towers. KataCR
trains several detectors, each specialised on a subset of classes, and merges
their outputs. `ComboDetector` does the same: each sub-model's class names are
mapped to global names and merged with cross-model NMS.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import ScreenConfig
from ..constants import SIDE_UNKNOWN, split_side
from ..regions import to_pixels
from .side import infer_side
from .types import Detection, nms


class YoloDetector:
    def __init__(self, weights: str | Path, conf: float = 0.35, iou: float = 0.6,
                 imgsz: int = 640, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.names: dict[int, str] = self.model.names
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device

    def __call__(self, img: np.ndarray, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        res = self.model.predict(img, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                 device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        ox, oy = offset
        out = []
        for b, c, k in zip(xyxy, conf, cls):
            name, side = split_side(self.names[k])  # "knight1" -> knight, enemy
            out.append(Detection((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy), float(c), name, side))
        return out


class ComboDetector:
    """Runs one or more detectors on the arena crop and merges the results."""

    def __init__(self, detectors: list, cfg: ScreenConfig, merge_iou: float = 0.6,
                 assign_sides: bool = True):
        self.detectors, self.cfg = detectors, cfg
        self.merge_iou, self.assign_sides = merge_iou, assign_sides

    @classmethod
    def from_weights(cls, weights: list[str | Path], cfg: ScreenConfig, **kw) -> "ComboDetector":
        det_kw = {k: kw.pop(k) for k in ("conf", "iou", "imgsz", "device") if k in kw}
        return cls([YoloDetector(w, **det_kw) for w in weights], cfg, **kw)

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = to_pixels(self.cfg.arena, w, h)
        arena = frame[y0:y1, x0:x1]
        dets: list[Detection] = []
        for d in self.detectors:
            dets.extend(d(arena, offset=(x0, y0)))
        dets = nms(dets, self.merge_iou) if len(self.detectors) > 1 else dets
        if self.assign_sides:
            for d in dets:
                if d.side == SIDE_UNKNOWN:  # model wasn't trained with side-tagged classes
                    d.side = infer_side(frame, d, h)
        return dets
