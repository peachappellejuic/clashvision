# clashcv

A Clash Royale computer-vision stack modelled on
[KataCR](https://github.com/vegetableleaf/KataCR). It takes only the phone screen
as input and produces a structured game state: units with team and tile
position, hand cards, elixir, tower HP, the cards you played, and a reward.
An agent can then act on that state through `adb`.

KataCR uses JAX and PaddleOCR. This project uses **PyTorch + Ultralytics
YOLOv8**, so it's easier to install and runs on CPU or GPU.

```
 phone ──scrcpy/adb──▶ frame
                         ├─ arena  ─▶ YOLOv8 combo detector ─▶ side (ally/enemy) ─▶ IoU tracker ─┐
                         ├─ hand   ─▶ card classifier (ResNet18 / templates)                   ├─▶ GameState
                         ├─ elixir ─▶ bar-fill reader                                          │   + Action
                         ├─ towers ─▶ HP bar-fill reader                                       │   + Reward
                         └─ timer  ─▶ OCR (Paddle/Easy/Tesseract)                              ┘
                                                                     │
                                       RuleAgent / Decision Transformer ─▶ adb tap (card, tile)
```

| KataCR | here |
|---|---|
| `build_dataset/generator.py` (synthetic data) | `clashcv/generation/generator.py` |
| `yolov8/` + `combo_detect.py` | `clashcv/detection/` (multi-model merge + tracker) |
| `classification/` ResNet cards / elixir | `clashcv/classification/cards.py`, `perception/gauges.py` |
| `ocr_text/` PaddleOCR | `perception/gauges.py::TimerOCR` |
| `policy/perceptron` SARBuilder | `perception/sar.py` (`Perceiver`, `ActionDetector`, `RewardTracker`) |
| `policy/offline` StARformer / DT | `policy/offline.py` (`build_episode`, `train_dt`) |
| `interact/` phone control | `capture.py`, `policy/control.py` |

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest            # 20 tests, all procedural (no game assets needed)
```

## Quick demo (no game files needed)

```bash
python -m clashcv demo
```

The first run draws fake sprites and cards and generates a synthetic dataset.
It fine-tunes YOLOv8n (about 6 minutes on a laptop CPU, faster on a GPU),
renders an animated phone-screen video, then runs the full pipeline in a
window. Press `q` to quit. Everything is cached in `runs/demo/`, so later runs
start immediately. Use `--rebuild` to start over. Without a display, it writes
`runs/demo/demo_overlay.mp4` instead.

## Labelled video (KataCR-style overlay)

```bash
python -m clashcv annotate match.mp4 labelled.mp4 --weights best.pt --cards weights/cards.pt
python -m clashcv annotate match.mp4 labelled.mp4 --weights best.pt --cards weights/cards.pt --layout panels
```

- **Units:** boxes in a colour per class, labelled `<class><side> <track-id> <conf>`,
  e.g. `knight0 4 0.97`. Side 0 is ally and 1 is enemy, as in KataCR.
- **Hand cards:** each slot is boxed with the card name on top and
  `<cost>e <confidence>` underneath. Green means you can afford it, red means
  you can't. The next card is boxed as well.
- **HUD:** Reward, Frame, Time and Elixir, plus a `PLAY <card> @ (x,y)` callout
  whenever you deploy a card.
- **`--layout panels`:** KataCR's eval view. The raw screen, the labelled
  screen and the 18×32 state grid sit side by side, and the tile of each played
  card is highlighted.

`run` takes the same `--layout` and `--height` options for live viewing
(`--show`).

## Workflow

No game art ships with this repo. Record your own gameplay and build the assets
from it.

**1. Capture.** Record matches (`scrcpy --record match.mp4`). For live input,
mirror into v4l2 the way KataCR does:
```bash
sudo modprobe v4l2loopback
scrcpy --v4l2-sink=/dev/video2 --no-video-playback --max-fps 30
```

**2. Calibrate screen regions.** Defaults are tuned for 1080×2400. Take one
screenshot and check the overlay:
```bash
python -m clashcv calibrate shot.png --out calib.png --dump my_screen.yaml
# edit my_screen.yaml until the boxes/grid line up, then pass --config my_screen.yaml everywhere
```

**3. Build the sprite library (the key KataCR idea).**
```bash
python -m clashcv extract-frames match.mp4 frames/ --every 30 --arena
# label a few hundred arena frames in CVAT/Label Studio/Roboflow (YOLO format)
python -m clashcv extract-segments frames/ labels/ assets/segments --names classes.txt
# review assets/segments/<class>/*.png and delete bad cut-outs
# put a few frames with an empty arena in assets/backgrounds/ (also arena crops)
```

**4. Generate a synthetic dataset and train the detector.**
```bash
python -m clashcv generate --n-train 20000 --n-val 1000 --out data/synth
python -m clashcv train-det data/synth/data.yaml --model yolov8s.pt --epochs 100
```
Sprites are drawn back to front by foot position, so occlusion looks like the
game's. Objects that are mostly hidden are left out of the labels. Mixing in
your hand-labelled real frames will help. For many classes, train several
specialised models and pass them all with `--weights a.pt b.pt` (combo).

**5. Card classifier.**
```bash
python -m clashcv extract-cards match.mp4 data/cards_raw --every 15
# sort crops into data/cards/<card-name>/ (include an "empty" class)
python -m clashcv train-cls data/cards --out weights/cards.pt
```
To recognise **every** card (not just the ones you've recorded), synthesise
hand crops from card art (`assets/card_art/<card>.png`, RGBA) and merge your
real crops in:
```bash
python -m clashcv synth-cards --art assets/card_art --real data/cards --out data/cards_full
python -m clashcv train-cls data/cards_full --out weights/cards.pt --epochs 10 --batch 64
```
To skip training, pass `--cards templates/`, a folder holding one
`<card-name>.png` per card.

**6. Run.**
```bash
python -m clashcv run /dev/video2 --weights runs/detect/clash/weights/best.pt \
    --cards weights/cards.pt --config my_screen.yaml --show
# --save out.mp4 writes an overlay video; --ocr reads the clock
# --play [--dry-run] lets the rule agent play through adb
```

**7. (Optional) Learned policy, offline, like KataCR.**
```bash
python -m clashcv build-episode replay1.mp4 eps/ep1.npz --weights best.pt --cards weights/cards.pt --classes data/synth/classes.json
python -m clashcv train-policy eps/ --classes data/synth/classes.json
```

## Design notes

- **Ally vs enemy.** KataCR adds a "belong" head to YOLO. Here the team comes
  from the colour of the HP bar or level badge above each box (blue or red).
  If no bar is visible, the unit's half of the arena decides. So one class per
  unit type is enough.
- **Elixir and tower HP** are read from bar fill (rightmost filled column),
  not OCR. This is faster and isn't thrown off by the digits drawn on the bar.
- **Action detection.** A hand slot changes, elixir drops by about the card's
  cost, and a new ally track of the matching unit appears. That track's foot
  position gives the deploy tile. `CARD_TO_UNITS` in `constants.py` maps cards
  to the units they spawn.
- **Coordinates.** Everything is in relative screen boxes and an 18×32 tile
  grid, so any resolution with the same aspect ratio works after calibration.

Use this for research and offline analysis. Automating play in live ladder
matches is against Supercell's Terms of Service. Use friendly battles or
training camp.
