"""State / Action / Reward builder (KataCR's `perceptron` + `SARBuilder`).

Fuses detector, tracker, card classifier and gauge readers into a `GameState`
per frame, infers which card *you* played (and where) by watching the hand and
new ally units, and computes a reward from tower damage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import ScreenConfig
from ..constants import (ELIXIR_COST, EMPTY_CARD, SIDE_ALLY, SIDE_ENEMY, TOWER_CLASSES,
                         units_for_card)
from ..detection.tracker import IoUTracker
from ..detection.types import Detection
from ..regions import Grid, crop
from .gauges import TimerOCR, read_elixir, read_tower_hp


@dataclass
class Unit:
    cls: str
    side: int
    tile: tuple[float, float]
    xyxy: tuple[float, float, float, float]
    track_id: int
    conf: float


@dataclass
class GameState:
    frame_idx: int
    time_left: int | None
    elixir: float
    hand: list[str]
    hand_conf: list[float]
    units: list[Unit]
    tower_hp: dict[str, float | None]
    towers_alive: dict[int, int] = field(default_factory=dict)  # side -> count
    next_card: str | None = None

    def ally(self) -> list[Unit]:
        return [u for u in self.units if u.side == SIDE_ALLY and u.cls not in TOWER_CLASSES]

    def enemy(self) -> list[Unit]:
        return [u for u in self.units if u.side == SIDE_ENEMY and u.cls not in TOWER_CLASSES]

    def to_arrays(self, class_names: list[str], grid_size=(18, 32)) -> dict[str, np.ndarray]:
        """Policy-ready features: class-id and side maps over the tile grid + scalars."""
        cols, rows = grid_size
        idx = {c: i + 1 for i, c in enumerate(class_names)}
        cls_map = np.zeros((rows, cols), np.int16)
        side_map = np.zeros((rows, cols), np.int8)
        for u in self.units:
            c, r = int(np.clip(u.tile[0], 0, cols - 1)), int(np.clip(u.tile[1], 0, rows - 1))
            cls_map[r, c] = idx.get(u.cls, 0)
            side_map[r, c] = 1 if u.side == SIDE_ALLY else -1
        card_idx = {c: i + 1 for i, c in enumerate(sorted(ELIXIR_COST))}
        return {
            "cls_map": cls_map,
            "side_map": side_map,
            "hand": np.array([card_idx.get(c, 0) for c in self.hand], np.int16),
            "scalars": np.array([self.elixir / 10.0, (self.time_left or 0) / 180.0], np.float32),
        }


@dataclass
class Action:
    frame_idx: int
    card: str
    slot: int
    tile: tuple[int, int] | None  # None if the deploy position couldn't be localised


class ActionDetector:
    """Detects the player's own card plays from hand + elixir + new ally tracks."""

    def __init__(self, resolve_frames: int = 15):
        self.prev_hand: list[str] | None = None
        self.prev_elixir: float | None = None
        self.pending: list[tuple[Action, set[int]]] = []  # action, track ids seen at play time
        self.resolve_frames = resolve_frames

    def update(self, st: GameState) -> list[Action]:
        out: list[Action] = []
        seen_ids = {u.track_id for u in st.units}
        if self.prev_hand is not None:
            for slot, (old, new) in enumerate(zip(self.prev_hand, st.hand)):
                if old in (EMPTY_CARD, new) or old not in ELIXIR_COST:
                    continue
                # Elixir must have dropped (allow regeneration noise of ~1).
                drop = (self.prev_elixir or 0) - st.elixir
                if drop >= ELIXIR_COST[old] - 1.2:
                    self.pending.append((Action(st.frame_idx, old, slot, None), set(seen_ids)))
        still = []
        for act, known in self.pending:
            targets = set(units_for_card(act.card))
            fresh = [u for u in st.units
                     if u.side == SIDE_ALLY and u.cls in targets and u.track_id not in known]
            if fresh:
                tx = float(np.mean([u.tile[0] for u in fresh]))
                ty = float(np.max([u.tile[1] for u in fresh]))  # deploy point ~ front-most foot
                act.tile = (int(tx), int(ty))
                out.append(act)
            elif st.frame_idx - act.frame_idx >= self.resolve_frames:
                out.append(act)  # unresolved position
            else:
                still.append((act, known))
        self.pending = still
        self.prev_hand, self.prev_elixir = list(st.hand), st.elixir
        return out


class RewardTracker:
    """r = enemy tower HP lost - ally tower HP lost (fractions), ±1 per tower destroyed."""

    def __init__(self, destroy_bonus: float = 1.0, confirm_frames: int = 5):
        self.last_hp: dict[str, float] = {}
        self.last_alive: dict[int, int] = {}
        self.low_streak: dict[int, int] = {}
        self.destroy_bonus, self.confirm_frames = destroy_bonus, confirm_frames

    def update(self, st: GameState) -> float:
        r = 0.0
        for name, hp in st.tower_hp.items():
            if hp is None:
                continue
            prev = self.last_hp.get(name, 1.0)
            if hp < prev:  # HP only goes down; ignore upward read noise
                r += (prev - hp) * (1 if name.startswith("enemy") else -1)
                self.last_hp[name] = hp
        for side, n in st.towers_alive.items():
            prev = self.last_alive.get(side)
            if prev is None:
                self.last_alive[side] = n
                continue
            # Towers can flicker out of detection: require a few consecutive low counts.
            self.low_streak[side] = self.low_streak.get(side, 0) + 1 if n < prev else 0
            if self.low_streak[side] >= self.confirm_frames:
                r += self.destroy_bonus * (prev - n) * (1 if side == SIDE_ENEMY else -1)
                self.last_alive[side], self.low_streak[side] = n, 0
        return r


class Perceiver:
    """Frame -> GameState, plus actions and reward, using whichever components exist."""

    def __init__(self, cfg: ScreenConfig, detector=None, card_classifier=None,
                 tracker: IoUTracker | None = None, ocr: TimerOCR | None = None):
        self.cfg, self.detector, self.cards, self.ocr = cfg, detector, card_classifier, ocr
        self.tracker = tracker or IoUTracker()
        self.actions = ActionDetector()
        self.reward = RewardTracker()
        self.frame_idx = 0
        self._card_thumbs: list[np.ndarray | None] = []
        self._card_preds: list[tuple[str, float]] = []
        self._card_age = 0

    def reset(self) -> None:
        self.tracker.reset()
        self.actions = ActionDetector()
        self.reward = RewardTracker()
        self.frame_idx = 0
        self._card_thumbs, self._card_preds, self._card_age = [], [], 0

    # A slot is re-classified only when its crop visibly changes (card played, or
    # greyed out / back to affordable). Cards are static otherwise, so this skips the
    # classifier on almost every frame. Everything is refreshed every `CARD_REFRESH` frames.
    CARD_DIFF = 6.0      # mean abs grey-level change of the 24x30 thumbnail
    CARD_REFRESH = 60

    def _classify_cards(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        thumbs = [cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (24, 30),
                             interpolation=cv2.INTER_AREA).astype(np.float32) for c in crops]
        self._card_age += 1
        if len(self._card_preds) != len(crops) or self._card_age >= self.CARD_REFRESH:
            changed = list(range(len(crops)))
            self._card_age = 0
        else:
            changed = [i for i, (t, old) in enumerate(zip(thumbs, self._card_thumbs))
                       if float(np.abs(t - old).mean()) > self.CARD_DIFF]
        if changed:
            preds = self.cards.predict([crops[i] for i in changed])
            if len(self._card_preds) != len(crops):
                self._card_preds = [("", 0.0)] * len(crops)
                self._card_thumbs = [None] * len(crops)
            for i, p in zip(changed, preds):
                self._card_preds[i], self._card_thumbs[i] = p, thumbs[i]
        return list(self._card_preds)

    def state(self, frame: np.ndarray, dets: list[Detection] | None = None) -> GameState:
        h, w = frame.shape[:2]
        grid = Grid(self.cfg, w, h)
        if dets is None:
            dets = self.detector(frame) if self.detector else []
        dets = self.tracker.update(dets)
        units = [Unit(d.cls, d.side, grid.pixel_to_tile(*d.foot), d.xyxy, d.track_id, d.conf)
                 for d in dets]
        alive = {SIDE_ALLY: 0, SIDE_ENEMY: 0}
        for u in units:
            if u.cls in TOWER_CLASSES and u.side in alive:
                alive[u.side] += 1
        next_card = None
        if self.cards:
            preds = self._classify_cards([crop(frame, b) for b in [*self.cfg.cards, self.cfg.next_card]])
            hand, conf = [p[0] for p in preds[:4]], [p[1] for p in preds[:4]]
            next_card = preds[4][0]
        else:
            hand, conf = [EMPTY_CARD] * 4, [0.0] * 4
        st = GameState(
            frame_idx=self.frame_idx,
            time_left=self.ocr.seconds_left(frame, self.cfg) if self.ocr else None,
            elixir=read_elixir(frame, self.cfg),
            hand=hand, hand_conf=conf, units=units,
            tower_hp=read_tower_hp(frame, self.cfg),
            towers_alive=alive if any(alive.values()) else {},
            next_card=next_card,
        )
        self.frame_idx += 1
        return st

    def step(self, frame: np.ndarray, dets: list[Detection] | None = None):
        """Returns (state, actions_resolved_this_frame, reward)."""
        st = self.state(frame, dets)
        return st, self.actions.update(st), self.reward.update(st)
