"""Frame sources: video files, image folders, scrcpy v4l2 devices, adb, desktop.

The fastest live setup on Linux (same as KataCR) is mirroring the phone with
scrcpy into a v4l2 loopback device:

    sudo modprobe v4l2loopback
    scrcpy --v4l2-sink=/dev/video2 --no-video-playback --max-fps 30

and then `open_source("/dev/video2")`.
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class FrameSource(Protocol):
    def __iter__(self) -> Iterator[np.ndarray]: ...


class VideoSource:
    """Video file, v4l2 device path (/dev/videoN) or integer camera index."""

    def __init__(self, src: str | int, stride: int = 1):
        self.src, self.stride = src, max(1, stride)

    def __iter__(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video source {self.src!r}")
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % self.stride == 0:
                    yield frame
                i += 1
        finally:
            cap.release()


class ImageFolderSource:
    def __init__(self, folder: str | Path):
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)

    def __iter__(self):
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is not None:
                yield img


class AdbScreencap:
    """Slow (~2-5 fps) but dependency-free: `adb exec-out screencap -p`."""

    def __init__(self, serial: str | None = None):
        self.cmd = ["adb"] + (["-s", serial] if serial else []) + ["exec-out", "screencap", "-p"]

    def grab(self) -> np.ndarray:
        png = subprocess.run(self.cmd, capture_output=True, check=True).stdout
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("adb screencap returned no image")
        return img

    def __iter__(self):
        while True:
            yield self.grab()


class ScreenRegionSource:
    """Grab a region of the desktop (e.g. a scrcpy window). Requires `mss`."""

    def __init__(self, left: int, top: int, width: int, height: int):
        self.mon = {"left": left, "top": top, "width": width, "height": height}

    def __iter__(self):
        import mss

        with mss.mss() as sct:
            while True:
                yield cv2.cvtColor(np.asarray(sct.grab(self.mon)), cv2.COLOR_BGRA2BGR)


class LatestFrame:
    """Reads a live source on a background thread and yields only the newest frame.

    Without this, frames queue up (in OpenCV's/v4l2's buffer, or while adb blocks)
    whenever processing is slower than capture, and the output drifts further and
    further behind the game. Here capture and processing overlap, and stale frames
    are dropped, so what you see is always the current screen.
    """

    def __init__(self, source: FrameSource):
        self.source = source

    def __iter__(self):
        cond = threading.Condition()
        state = {"frame": None, "seq": 0, "done": False, "error": None}

        def reader():
            try:
                for f in self.source:
                    with cond:
                        state["frame"], state["seq"] = f, state["seq"] + 1
                        cond.notify()
                        if state["done"]:
                            return
            except Exception as e:  # surfaced in the consumer
                state["error"] = e
            finally:
                with cond:
                    state["done"] = True
                    cond.notify()

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        last = 0
        try:
            while True:
                with cond:
                    cond.wait_for(lambda: state["seq"] != last or state["done"])
                    if state["seq"] == last:  # source ended with nothing new
                        break
                    frame, last = state["frame"], state["seq"]
                yield frame
            if state["error"] is not None:
                raise state["error"]
        finally:
            with cond:
                state["done"] = True


def is_live(spec: str) -> bool:
    return (spec.startswith(("adb", "screen:", "/dev/video")) or spec.isdigit())


def open_source(spec: str, stride: int = 1, latest: bool | None = None) -> FrameSource:
    """`adb[:serial]`, `screen:x,y,w,h`, a folder, a video/device path, or a camera index.

    `latest` (default: on for live sources) drops stale frames; see `LatestFrame`.
    """
    if spec.startswith("adb"):
        src = AdbScreencap(spec.split(":", 1)[1] if ":" in spec else None)
    elif spec.startswith("screen:"):
        src = ScreenRegionSource(*map(int, spec[7:].split(",")))
    elif spec.isdigit():
        src = VideoSource(int(spec), stride)
    elif Path(spec).is_dir():
        src = ImageFolderSource(spec)
    else:
        src = VideoSource(spec, stride)
    if latest is None:
        latest = is_live(spec)
    return LatestFrame(src) if latest else src
