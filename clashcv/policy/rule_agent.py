"""A transparent rule-based agent - a baseline before training a learned policy.

Defend: when an enemy ground unit crosses into our half, drop the cheapest
affordable troop just in front of it, on our side.
Attack: at (near) full elixir, play the most expensive affordable card at the
bridge of the lane with fewer enemies.
"""
from __future__ import annotations

from ..constants import ELIXIR_COST, EMPTY_CARD, SPELL_CARDS
from ..perception.sar import GameState

RIVER_ROW = 15.5  # rows 0-14 enemy half, 17-31 our half (18x32 grid)
BRIDGE_TILES = {"left": (3, 17), "right": (14, 17)}


class RuleAgent:
    def __init__(self, cooldown_frames: int = 20, full_elixir: float = 9.0):
        self.cooldown_frames, self.full_elixir = cooldown_frames, full_elixir
        self.last_play = -10**9

    def _affordable(self, st: GameState, troops_only: bool = False) -> list[tuple[int, str]]:
        out = []
        for i, c in enumerate(st.hand):
            if c == EMPTY_CARD or c not in ELIXIR_COST or ELIXIR_COST[c] > st.elixir:
                continue
            if troops_only and c in SPELL_CARDS:
                continue
            out.append((i, c))
        return out

    def act(self, st: GameState) -> tuple[int, tuple[int, int]] | None:
        if st.frame_idx - self.last_play < self.cooldown_frames:
            return None
        threats = [u for u in st.enemy() if u.tile[1] > RIVER_ROW]
        if threats:
            cands = self._affordable(st, troops_only=True)
            if cands:
                tgt = max(threats, key=lambda u: u.tile[1])  # deepest threat
                slot, _ = min(cands, key=lambda ic: ELIXIR_COST[ic[1]])
                tile = (int(tgt.tile[0]), int(min(31, tgt.tile[1] + 2)))
                self.last_play = st.frame_idx
                return slot, tile
        if st.elixir >= self.full_elixir:
            cands = self._affordable(st, troops_only=True)
            if cands:
                left = sum(1 for u in st.enemy() if u.tile[0] < 9)
                right = len(st.enemy()) - left
                lane = "left" if left <= right else "right"
                slot, _ = max(cands, key=lambda ic: ELIXIR_COST[ic[1]])
                self.last_play = st.frame_idx
                return slot, BRIDGE_TILES[lane]
        return None
