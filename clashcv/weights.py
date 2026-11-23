"""Download the pretrained models from the GitHub release."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

RELEASE = "v0.1.0"
BASE_URL = f"https://github.com/peachappellejuic/clashvision/releases/download/{RELEASE}"

# file -> (sha256, description)
WEIGHTS = {
    "best.pt": ("40d3006c8e962127d17a16e1ee917f05f20fef982bb92b60f62a0baf7d446ba9",
                "YOLOv8 arena detector, 212 classes (KataCR dataset)"),
    "cards.pt": ("d6291ed57e6fd99584b7fd2cad399c1cd1b880a4263f8d1c680bcc396bd412fb",
                 "ResNet18 hand-card classifier, 178 classes"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(out: str | Path = "weights", force: bool = False) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, (sha, desc) in WEIGHTS.items():
        dst = out / name
        if dst.exists() and not force and _sha256(dst) == sha:
            print(f"{dst} already up to date")
            paths.append(dst)
            continue
        url = f"{BASE_URL}/{name}"
        print(f"downloading {name} ({desc})\n  {url}")
        tmp = dst.with_suffix(dst.suffix + ".part")
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink()
            raise SystemExit(f"checksum mismatch for {name}: expected {sha}, got {got}")
        tmp.replace(dst)
        print(f"  saved {dst}")
        paths.append(dst)
    return paths
