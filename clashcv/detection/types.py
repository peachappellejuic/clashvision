from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    xyxy: tuple[float, float, float, float]  # full-frame pixels
    conf: float
    cls: str
    side: int = -1  # constants.SIDE_*
    track_id: int = -1

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre: where the unit stands on the ground grid."""
        x0, _, x1, y1 = self.xyxy
        return (x0 + x1) / 2, y1


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def nms(dets: list[Detection], iou_thr: float = 0.6, class_agnostic: bool = False) -> list[Detection]:
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i].conf)
    boxes = np.array([dets[i].xyxy for i in order], dtype=np.float32)
    ious = iou_matrix(boxes, boxes)
    keep, suppressed = [], np.zeros(len(order), bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        keep.append(dets[order[i]])
        for j in range(i + 1, len(order)):
            if ious[i, j] > iou_thr and (class_agnostic or dets[order[i]].cls == dets[order[j]].cls):
                suppressed[j] = True
    return keep
