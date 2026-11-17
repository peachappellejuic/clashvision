import numpy as np
import pytest

torch = pytest.importorskip("torch")

from clashcv.policy.offline import COLS, ROWS, make_model, train_dt


def _fake_episode(path, T=40, n_classes=5, seed=0):
    rng = np.random.default_rng(seed)
    slot = np.zeros(T, np.int64)
    tile = np.zeros(T, np.int64)
    slot[::7] = rng.integers(1, 5, len(slot[::7]))
    tile[::7] = rng.integers(0, COLS * ROWS, len(tile[::7]))
    np.savez(path, cls_map=rng.integers(0, n_classes + 1, (T, ROWS, COLS)).astype(np.int16),
             side_map=rng.integers(-1, 2, (T, ROWS, COLS)).astype(np.int8),
             hand=rng.integers(0, 30, (T, 4)).astype(np.int16),
             scalars=rng.random((T, 2)).astype(np.float32),
             slot=slot, tile=tile, reward=rng.normal(0, 0.1, T).astype(np.float32))


def test_dt_forward_shapes():
    m = make_model(n_classes=5, context=8)
    B, T = 2, 8
    out_s, out_t = m(torch.zeros(B, T), torch.zeros(B, T, ROWS, COLS, dtype=torch.long),
                     torch.zeros(B, T, ROWS, COLS), torch.zeros(B, T, 4, dtype=torch.long),
                     torch.zeros(B, T, 2), torch.zeros(B, T, dtype=torch.long),
                     torch.zeros(B, T, dtype=torch.long))
    assert out_s.shape == (B, T, 5) and out_t.shape == (B, T, COLS * ROWS)


def test_train_dt_runs(tmp_path):
    for i in range(2):
        _fake_episode(tmp_path / f"ep{i}.npz", seed=i)
    train_dt(tmp_path, n_classes=5, out=tmp_path / "p.pt", steps=3, batch=2, context=10)
    assert (tmp_path / "p.pt").exists()
