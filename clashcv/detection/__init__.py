from .types import Detection, iou_matrix, nms
from .tracker import IoUTracker
from .side import infer_side

__all__ = ["Detection", "iou_matrix", "nms", "IoUTracker", "infer_side",
           "YoloDetector", "ComboDetector"]


def __getattr__(name):  # lazy: avoid importing ultralytics unless needed
    if name in ("YoloDetector", "ComboDetector"):
        from . import detector

        return getattr(detector, name)
    raise AttributeError(name)
