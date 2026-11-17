"""Offline imitation / return-conditioned policy (KataCR's Decision Transformer).

1. `build_episode` runs the Perceiver over a replay video of *your own* games
   and records (state arrays, action, reward) per frame -> .npz.
2. `DTPolicy` is a compact Decision Transformer: each timestep is
   [return-to-go, state, action] tokens; the state is encoded by a small CNN
   over the (class embedding + side) arena grid plus hand/elixir features.
   It predicts: no-op vs card slot (5-way) and the deploy tile (18*32-way).
3. `train_dt` trains it on a folder of episodes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COLS, ROWS = 18, 32
N_TILES = COLS * ROWS


def build_episode(frames, perceiver, class_names: list[str], out_path: str | Path) -> int:
    rec = {k: [] for k in ("cls_map", "side_map", "hand", "scalars", "slot", "tile", "reward")}
    perceiver.reset()  # so state.frame_idx == index into rec
    for frame in frames:
        st, actions, r = perceiver.step(frame)
        arr = st.to_arrays(class_names)
        for k in ("cls_map", "side_map", "hand", "scalars"):
            rec[k].append(arr[k])
        rec["slot"].append(0)
        rec["tile"].append(0)
        rec["reward"].append(r)
        for a in actions:  # actions resolve a few frames late: label the play frame
            if a.tile is None:
                continue
            t = a.frame_idx
            rec["slot"][t] = a.slot + 1
            rec["tile"][t] = a.tile[1] * COLS + a.tile[0]
    n = len(rec["slot"])
    if n:
        np.savez_compressed(out_path, **{k: np.asarray(v) for k, v in rec.items()})
    return n


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def make_model(n_classes: int, n_cards: int = 64, d: int = 128, n_layers: int = 4,
               n_heads: int = 4, context: int = 30):
    torch, nn = _torch()

    class StateEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.cls_emb = nn.Embedding(n_classes + 1, 16, padding_idx=0)
            self.cnn = nn.Sequential(
                nn.Conv2d(17, 32, 3, padding=1), nn.GELU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(), nn.Linear(64 * 12, d))
            self.card_emb = nn.Embedding(n_cards, d, padding_idx=0)
            self.scalar = nn.Linear(2, d)

        def forward(self, cls_map, side_map, hand, scalars):
            B, T = cls_map.shape[:2]
            x = self.cls_emb(cls_map.long().flatten(0, 1))  # (BT, R, C, 16)
            x = torch.cat([x, side_map.float().flatten(0, 1)[..., None]], -1).permute(0, 3, 1, 2)
            g = self.cnn(x).view(B, T, d)
            return g + self.card_emb(hand.long()).sum(2) + self.scalar(scalars)

    class DT(nn.Module):
        def __init__(self):
            super().__init__()
            self.context = context
            self.state = StateEncoder()
            self.rtg = nn.Linear(1, d)
            self.act = nn.Embedding(5 * N_TILES, d)
            self.pos = nn.Parameter(torch.zeros(1, 3 * context, d))
            layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, 0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, n_layers)
            self.head_slot = nn.Linear(d, 5)
            self.head_tile = nn.Linear(d, N_TILES)

        def forward(self, rtg, cls_map, side_map, hand, scalars, slot, tile):
            B, T = rtg.shape[:2]
            s = self.state(cls_map, side_map, hand, scalars)
            a = self.act(slot.long() * N_TILES + tile.long())
            tok = torch.stack([self.rtg(rtg[..., None]), s, a], 2).reshape(B, 3 * T, d) + self.pos[:, :3 * T]
            mask = torch.triu(torch.ones(3 * T, 3 * T, dtype=torch.bool, device=tok.device), 1)
            h = self.tf(tok, mask=mask)[:, 1::3]  # predict action from the state token
            return self.head_slot(h), self.head_tile(h)

    return DT()


def _load_episodes(folder: str | Path) -> list[dict[str, np.ndarray]]:
    eps = []
    for p in sorted(Path(folder).glob("*.npz")):
        e = dict(np.load(p))
        e["rtg"] = np.cumsum(e["reward"][::-1])[::-1].astype(np.float32)
        eps.append(e)
    if not eps:
        raise FileNotFoundError(f"no episodes (*.npz) in {folder}")
    return eps


def train_dt(episodes_dir: str | Path, n_classes: int, out: str | Path, steps: int = 2000,
             batch: int = 16, context: int = 30, lr: float = 3e-4, device: str = "cpu",
             noop_weight: float = 0.05):
    torch, nn = _torch()
    eps = [e for e in _load_episodes(episodes_dir) if len(e["slot"]) >= context]
    if not eps:
        raise ValueError(f"all episodes shorter than context={context}")
    model = make_model(n_classes, context=context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Most frames are no-ops: down-weight that class so the model learns to act.
    slot_w = torch.tensor([noop_weight, 1, 1, 1, 1], dtype=torch.float32, device=device)
    rng = np.random.default_rng(0)
    keys = ("rtg", "cls_map", "side_map", "hand", "scalars", "slot", "tile")
    for step in range(steps):
        bt = {k: [] for k in keys}
        for _ in range(batch):
            e = eps[rng.integers(len(eps))]
            i = rng.integers(0, len(e["slot"]) - context + 1)
            for k in keys:
                bt[k].append(e[k][i:i + context])
        bt = {k: torch.as_tensor(np.stack(v)).to(device) for k, v in bt.items()}
        # Teacher forcing: the action token at t is the *previous* action.
        prev_slot = torch.roll(bt["slot"], 1, 1); prev_slot[:, 0] = 0
        prev_tile = torch.roll(bt["tile"], 1, 1); prev_tile[:, 0] = 0
        ls, lt = model(bt["rtg"].float(), bt["cls_map"], bt["side_map"], bt["hand"],
                       bt["scalars"].float(), prev_slot, prev_tile)
        loss = nn.functional.cross_entropy(ls.flatten(0, 1), bt["slot"].long().flatten(), weight=slot_w)
        acted = bt["slot"].flatten() > 0
        if acted.any():
            loss = loss + nn.functional.cross_entropy(lt.flatten(0, 1)[acted], bt["tile"].long().flatten()[acted])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"step {step}  loss {loss.item():.4f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "n_classes": n_classes, "context": context}, out)
    return model
