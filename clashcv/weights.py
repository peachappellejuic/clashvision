"""Download the pretrained models from the GitHub release."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = "peachappellejuic/clashvision"
RELEASE = "v0.1.0"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE}"
API_URL = f"https://api.github.com/repos/{REPO}"

# file -> (sha256, description)
WEIGHTS = {
    "best.pt": ("40d3006c8e962127d17a16e1ee917f05f20fef982bb92b60f62a0baf7d446ba9",
                "YOLOv8 arena detector, 212 classes (KataCR dataset)"),
    "cards.pt": ("d6291ed57e6fd99584b7fd2cad399c1cd1b880a4263f8d1c680bcc396bd412fb",
                 "ResNet18 hand-card classifier, 178 classes"),
    # unzipped to weights/best_int8_openvino_model/ - ~3.5x faster detector on CPU
    "best_int8_openvino_model.zip": ("605b4ccdf23043797e106cec16fbfe451127c68f471fae2ade2bf271c0aebcf7",
                                     "OpenVINO INT8 detector for fast live inference"),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _token() -> str | None:
    """GitHub token for private repos: $GITHUB_TOKEN, $GH_TOKEN, or `gh auth token`."""
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _request(url: str, token: str | None, accept: str) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "clashcv"})
    if token:
        # unredirected: the asset download redirects to a signed storage URL that
        # rejects (and must not receive) the GitHub token
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def _asset_urls(token: str) -> dict[str, str]:
    """name -> API download URL for each asset of the release (works for private repos)."""
    req = _request(f"{API_URL}/releases/tags/{RELEASE}", token, "application/vnd.github+json")
    with urllib.request.urlopen(req) as r:
        release = json.load(r)
    return {a["name"]: a["url"] for a in release.get("assets", [])}


def download(out: str | Path = "weights", force: bool = False) -> list[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    token = _token()
    assets = None
    paths = []
    for name, (sha, desc) in WEIGHTS.items():
        dst = out / name
        if dst.exists() and not force and _sha256(dst) == sha:
            print(f"{dst} already up to date")
            paths.append(_unpack(dst))
            continue
        try:
            if token:
                if assets is None:
                    assets = _asset_urls(token)
                if name not in assets:
                    raise SystemExit(f"release {RELEASE} of {REPO} has no asset {name}")
                url, accept = assets[name], "application/octet-stream"
            else:
                url, accept = f"{BASE_URL}/{name}", "*/*"
            print(f"downloading {name} ({desc})")
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(_request(url, token, accept)) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                hint = ("check that the token can read the repo" if token else
                        "if the repo is private, set GITHUB_TOKEN (or log in with `gh auth login`)")
                raise SystemExit(f"cannot download {name} from {REPO} release {RELEASE} "
                                 f"(HTTP {e.code}): {hint}") from None
            raise
        got = _sha256(tmp)
        if got != sha:
            tmp.unlink()
            raise SystemExit(f"checksum mismatch for {name}: expected {sha}, got {got}")
        tmp.replace(dst)
        print(f"  saved {dst}")
        paths.append(_unpack(dst, force=True))
    return paths


def _unpack(path: Path, force: bool = False) -> Path:
    """Extract model-folder zips next to themselves; returns the usable path."""
    if path.suffix != ".zip":
        return path
    folder = path.with_suffix("")
    if force or not folder.is_dir():
        with zipfile.ZipFile(path) as z:
            for m in z.namelist():  # refuse paths escaping the weights folder
                if Path(m).is_absolute() or ".." in Path(m).parts:
                    raise SystemExit(f"unsafe path in {path}: {m}")
            z.extractall(path.parent)
        print(f"  extracted {folder}")
    return folder
