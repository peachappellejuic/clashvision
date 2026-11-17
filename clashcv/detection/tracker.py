"""Minimal IoU + class tracker so units keep stable ids across frames.

Works on merged ComboDetector output (Ultralytics' built-in ByteTrack only
tracks a single model). Greedy matching on IoU, same class and same side.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Detection, iou_matrix


@dataclass
class _Track:
    id: int
    det: Detection
    misses: int = 0
    hits: int = 1


class IoUTracker:
    def __init__(self, iou_thr: float = 0.25, max_misses: int = 8, min_hits: int = 1):
        self.iou_thr, self.max_misses, self.min_hits = iou_thr, max_misses, min_hits
        self.tracks: list[_Track] = []
        self._next = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next = 1

    def update(self, dets: list[Detection]) -> list[Detection]:
        tb = np.array([t.det.xyxy for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        db = np.array([d.xyxy for d in dets], dtype=np.float32).reshape(-1, 4)
        ious = iou_matrix(tb, db)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(dets):
                if t.det.cls != d.cls or (t.det.side != d.side and -1 not in (t.det.side, d.side)):
                    ious[i, j] = 0.0
        matched_t, matched_d = set(), set()
        while ious.size and ious.max() >= self.iou_thr:
            i, j = np.unravel_index(int(ious.argmax()), ious.shape)
            t, d = self.tracks[i], dets[j]
            d.track_id = t.id
            t.det, t.misses, t.hits = d, 0, t.hits + 1
            matched_t.add(i)
            matched_d.add(j)
            ious[i, :] = -1
            ious[:, j] = -1
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        for j, d in enumerate(dets):
            if j not in matched_d:
                d.track_id = self._next
                self.tracks.append(_Track(self._next, d))
                self._next += 1
        hits = {t.id: t.hits for t in self.tracks}
        return [d for d in dets if hits.get(d.track_id, 0) >= self.min_hits]
