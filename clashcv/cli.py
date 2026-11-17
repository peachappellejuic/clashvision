"""`python -m clashcv <command>` - see README for the full workflow."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from .config import ScreenConfig


def _cfg(a) -> ScreenConfig:
    return ScreenConfig.load(getattr(a, "config", None))


def cmd_calibrate(a):
    cfg = _cfg(a)
    img = cv2.imread(a.frame)
    if img is None:
        raise SystemExit(f"cannot read {a.frame}")
    from .visualize import draw_regions

    cv2.imwrite(a.out, draw_regions(img, cfg))
    print(f"wrote overlay to {a.out}  (frame {img.shape[1]}x{img.shape[0]})")
    if a.dump:
        cfg.save(a.dump)
        print(f"wrote editable config to {a.dump}")


def cmd_extract_frames(a):
    from .capture import VideoSource
    from .regions import crop

    cfg, out = _cfg(a), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, f in enumerate(VideoSource(a.video, stride=a.every)):
        cv2.imwrite(str(out / f"{Path(a.video).stem}_{i:06d}.jpg"), crop(f, cfg.arena) if a.arena else f)
        n += 1
    print(f"saved {n} frames to {out}")


def cmd_extract_cards(a):
    """Dump hand-card crops for labelling into data/cards/<name>/."""
    from .capture import VideoSource
    from .regions import crop

    cfg, out = _cfg(a), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, f in enumerate(VideoSource(a.video, stride=a.every)):
        for s, b in enumerate(cfg.cards):
            cv2.imwrite(str(out / f"{i:06d}_s{s}.png"), crop(f, b))
            n += 1
    print(f"saved {n} card crops to {out} - sort them into {out}/<card-name>/ folders")


def cmd_extract_segments(a):
    from .generation import extract_segments

    names = [l.strip() for l in Path(a.names).read_text().splitlines() if l.strip()]
    print(f"extracted {extract_segments(a.images, a.labels, names, a.out)} segments into {a.out}")


def cmd_generate(a):
    from .constants import DETECTION_CLASSES
    from .generation import SyntheticGenerator, load_backgrounds, load_segments

    segs = load_segments(a.segments)
    names = sorted(segs) if a.names == "auto" else (
        DETECTION_CLASSES if a.names == "default" else json.loads(Path(a.names).read_text()))
    missing = set(segs) - set(names)
    if missing:
        raise SystemExit(f"segment classes not in class list: {sorted(missing)}")
    gen = SyntheticGenerator(segs, load_backgrounds(a.backgrounds), names, size=(a.width, a.height),
                             sprite_scale=a.scale, units_range=(a.min_units, a.max_units), seed=a.seed)
    y = gen.write_dataset(a.out, a.n_train, a.n_val)
    print(f"dataset written: {y}")


def cmd_train_det(a):
    from ultralytics import YOLO

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=a.project, name=a.name, plots=False,
                # Clash sprites have a meaningful orientation & colour (team!) - keep aug mild.
                fliplr=0.5, flipud=0.0, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, mosaic=0.5, degrees=0.0)


def cmd_train_cls(a):
    from .classification import train_resnet

    acc = train_resnet(a.data, a.out, epochs=a.epochs, batch=a.batch, device=a.device,
                       pretrained=not a.no_pretrained)
    print(f"saved {a.out}  (val acc {acc:.3f})")


def _perceiver(a, cfg):
    from .perception import Perceiver, TimerOCR

    det = None
    if a.weights:
        from .detection import ComboDetector

        det = ComboDetector.from_weights(a.weights, cfg, conf=a.conf, imgsz=a.imgsz, device=a.device)
    cards = None
    if a.cards:
        from .classification import ResNetCardClassifier, TemplateCardClassifier

        cards = TemplateCardClassifier(a.cards) if Path(a.cards).is_dir() else ResNetCardClassifier(a.cards)
    return Perceiver(cfg, det, cards, ocr=TimerOCR() if a.ocr else None)


def cmd_run(a):
    from .capture import open_source
    from .visualize import draw_state

    cfg = _cfg(a)
    per = _perceiver(a, cfg)
    agent = ctrl = None
    if a.play:
        from .policy import AdbController, RuleAgent

        agent = RuleAgent()
    writer = None
    t0, n = time.time(), 0
    for frame in open_source(a.source, a.stride):
        st, actions, r = per.step(frame)
        for act in actions:
            print(f"[{st.frame_idx}] played {act.card} (slot {act.slot}) at tile {act.tile}")
        if agent:
            if ctrl is None:
                from .policy import AdbController

                ctrl = AdbController(cfg, frame.shape[1], frame.shape[0], a.serial, dry_run=a.dry_run)
            if (move := agent.act(st)) is not None:
                print(f"[{st.frame_idx}] agent: slot {move[0]} ({st.hand[move[0]]}) -> tile {move[1]}")
                ctrl.play(*move)
        if a.show or a.save:
            vis = draw_state(frame, st, actions, r)
            if a.save:
                if writer is None:
                    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                             (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if a.show:
                cv2.imshow("clashcv", cv2.resize(vis, None, fx=a.view_scale, fy=a.view_scale))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        n += 1
        if n % 30 == 0:
            print(f"{n} frames  {n / (time.time() - t0):.1f} fps  elixir {st.elixir:.1f}  "
                  f"units {len(st.units)}  hand {st.hand}")
    if writer:
        writer.release()


def cmd_build_episode(a):
    from .capture import VideoSource
    from .constants import DETECTION_CLASSES
    from .policy.offline import build_episode

    cfg = _cfg(a)
    names = json.loads(Path(a.classes).read_text()) if a.classes else DETECTION_CLASSES
    n = build_episode(VideoSource(a.video, a.stride), _perceiver(a, cfg), names, a.out)
    print(f"wrote {n} steps to {a.out}")


def cmd_train_policy(a):
    from .constants import DETECTION_CLASSES
    from .policy.offline import train_dt

    n = len(json.loads(Path(a.classes).read_text())) if a.classes else len(DETECTION_CLASSES)
    train_dt(a.episodes, n, a.out, steps=a.steps, batch=a.batch, context=a.context, device=a.device)
    print(f"saved {a.out}")


def cmd_demo(a):
    import os
    import sys

    from .demo import build

    p = build(a.out, epochs=a.epochs, device=a.device, rebuild=a.rebuild)
    show = not a.no_show and (sys.platform != "linux" or os.environ.get("DISPLAY")
                              or os.environ.get("WAYLAND_DISPLAY"))
    save = str(Path(a.out) / "demo_overlay.mp4") if not show else None
    args = ["run", str(p["video"]), "--weights", str(p["weights"]), "--cards", str(p["cards"]),
            "--view-scale", str(a.view_scale)]
    if a.device:
        args += ["--device", a.device]
    args += ["--show"] if show else ["--save", save]
    print("[demo] running perception" + (" - press q in the window to quit" if show else ""))
    main(args)
    if save:
        print(f"[demo] no display found; wrote overlay video to {save}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="clashcv", description="Clash Royale computer vision toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help):
        s = sub.add_parser(name, help=help)
        s.set_defaults(fn=fn)
        s.add_argument("--config", help="ScreenConfig YAML override")
        return s

    def perception_args(s):
        s.add_argument("--weights", nargs="*", default=[], help="one or more YOLO .pt (combo)")
        s.add_argument("--cards", help="ResNet .pt checkpoint or a folder of card templates")
        s.add_argument("--ocr", action="store_true", help="read the timer with OCR")
        s.add_argument("--conf", type=float, default=0.35)
        s.add_argument("--imgsz", type=int, default=640)
        s.add_argument("--device", default=None)
        s.add_argument("--stride", type=int, default=1, help="use every Nth frame")

    s = add("calibrate", cmd_calibrate, "draw screen regions/grid on a screenshot")
    s.add_argument("frame"); s.add_argument("--out", default="calibration.png")
    s.add_argument("--dump", help="also write the current config as YAML for editing")

    s = add("extract-frames", cmd_extract_frames, "dump frames (or arena crops) from a video")
    s.add_argument("video"); s.add_argument("out"); s.add_argument("--every", type=int, default=30)
    s.add_argument("--arena", action="store_true", help="save only the arena crop")

    s = add("extract-cards", cmd_extract_cards, "dump hand-card crops for classifier labelling")
    s.add_argument("video"); s.add_argument("out"); s.add_argument("--every", type=int, default=30)

    s = add("extract-segments", cmd_extract_segments, "GrabCut sprites from YOLO-labelled frames")
    s.add_argument("images"); s.add_argument("labels"); s.add_argument("out")
    s.add_argument("--names", required=True, help="text file, one class name per line (label order)")

    s = add("generate", cmd_generate, "synthesise a YOLO dataset from sprites + backgrounds")
    s.add_argument("--segments", default="assets/segments")
    s.add_argument("--backgrounds", default="assets/backgrounds")
    s.add_argument("--out", default="data/synth")
    s.add_argument("--names", default="auto", help="'auto' (segment folders), 'default', or JSON list file")
    s.add_argument("--n-train", type=int, default=5000); s.add_argument("--n-val", type=int, default=500)
    s.add_argument("--width", type=int, default=576); s.add_argument("--height", type=int, default=896)
    s.add_argument("--scale", type=float, default=1.0, help="sprite scale vs. background")
    s.add_argument("--min-units", type=int, default=4); s.add_argument("--max-units", type=int, default=25)
    s.add_argument("--seed", type=int, default=None)

    s = add("train-det", cmd_train_det, "train a YOLOv8 detector")
    s.add_argument("data"); s.add_argument("--model", default="yolov8s.pt")
    s.add_argument("--epochs", type=int, default=100); s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--batch", type=int, default=16); s.add_argument("--device", default=None)
    s.add_argument("--project", default="runs/detect"); s.add_argument("--name", default="clash")

    s = add("train-cls", cmd_train_cls, "train the ResNet hand-card classifier")
    s.add_argument("data"); s.add_argument("--out", default="weights/cards.pt")
    s.add_argument("--epochs", type=int, default=15); s.add_argument("--batch", type=int, default=32)
    s.add_argument("--device", default="cpu"); s.add_argument("--no-pretrained", action="store_true")

    s = add("run", cmd_run, "run perception (and optionally the agent) on a source")
    s.add_argument("source", help="video, folder, /dev/videoN, camera idx, adb[:serial], screen:x,y,w,h")
    perception_args(s)
    s.add_argument("--show", action="store_true"); s.add_argument("--save", help="write overlay mp4")
    s.add_argument("--fps", type=float, default=30); s.add_argument("--view-scale", type=float, default=0.4)
    s.add_argument("--play", action="store_true", help="let the rule agent play via adb")
    s.add_argument("--dry-run", action="store_true", help="agent decides but doesn't tap")
    s.add_argument("--serial")

    s = add("build-episode", cmd_build_episode, "replay video -> offline SAR episode (.npz)")
    s.add_argument("video"); s.add_argument("out"); perception_args(s)
    s.add_argument("--classes", help="JSON class list (classes.json from generate)")

    s = add("train-policy", cmd_train_policy, "train the Decision Transformer on episodes")
    s.add_argument("episodes"); s.add_argument("--out", default="weights/policy.pt")
    s.add_argument("--classes"); s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch", type=int, default=16); s.add_argument("--context", type=int, default=30)
    s.add_argument("--device", default="cpu")

    s = add("demo", cmd_demo, "self-contained demo: builds fake assets, trains, and runs (no game files needed)")
    s.add_argument("--out", default="runs/demo", help="cache folder (reused on later runs)")
    s.add_argument("--epochs", type=int, default=20)
    s.add_argument("--device", default=None, help="e.g. cpu, 0 (GPU), mps")
    s.add_argument("--rebuild", action="store_true", help="delete the cache and rebuild")
    s.add_argument("--no-show", action="store_true", help="save an overlay mp4 instead of a window")
    s.add_argument("--view-scale", type=float, default=0.4)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
