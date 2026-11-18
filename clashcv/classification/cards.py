"""Hand-card classifiers.

* `TemplateCardClassifier` - zero training. One reference crop per card in
  `templates/<card>.png`; matches with normalised cross-correlation. Good enough
  to bootstrap labels.
* `ResNetCardClassifier` - ResNet18 fine-tuned on `data/cards/<card>/*.png`
  (torchvision ImageFolder layout), like KataCR's ResNet card classifier.
  Robust to the "greyed out / not enough elixir" card state.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
INPUT_SIZE = (96, 120)  # (w, h) - cards are portrait


def _prep_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    # Normalise brightness so greyed-out (unaffordable) cards still match.
    return cv2.equalizeHist(g).astype(np.float32)


class TemplateCardClassifier:
    def __init__(self, template_dir: str | Path, min_score: float = 0.45):
        self.names, self.templates = [], []
        for p in sorted(Path(template_dir).iterdir()):
            if p.suffix.lower() in _IMG_EXTS and (img := cv2.imread(str(p))) is not None:
                self.names.append(p.stem)
                self.templates.append(_prep_gray(img))
        if not self.names:
            raise FileNotFoundError(f"no card templates in {template_dir}")
        self.min_score = min_score

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        out = []
        for c in crops:
            g = _prep_gray(c)
            scores = [float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED)[0, 0]) for t in self.templates]
            k = int(np.argmax(scores))
            out.append((self.names[k], scores[k]) if scores[k] >= self.min_score else ("empty", scores[k]))
        return out


def build_resnet(num_classes: int, pretrained: bool = True):
    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights

    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _to_tensor(crops: list[np.ndarray]):
    import torch

    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    arr = np.stack([
        (cv2.cvtColor(cv2.resize(c, INPUT_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
         .astype(np.float32) / 255.0 - mean) / std
        for c in crops
    ])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class ResNetCardClassifier:
    def __init__(self, ckpt: str | Path, device: str = "cpu"):
        import torch

        state = torch.load(ckpt, map_location=device)
        self.names: list[str] = state["classes"]
        self.model = build_resnet(len(self.names), pretrained=False)
        self.model.load_state_dict(state["model"])
        self.model.eval().to(device)
        self.device = device

    def predict(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        import torch

        if not crops:
            return []
        with torch.no_grad():
            p = torch.softmax(self.model(_to_tensor(crops).to(self.device)), dim=1).cpu().numpy()
        return [(self.names[int(i)], float(r[i])) for r, i in zip(p, p.argmax(1))]


def train_resnet(data_dir: str | Path, out: str | Path, epochs: int = 15, batch: int = 32,
                 lr: float = 1e-3, device: str = "cpu", val_frac: float = 0.15,
                 pretrained: bool = True) -> float:
    """Fine-tune ResNet18 on an ImageFolder of card crops. Returns val accuracy."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, random_split

    root = Path(data_dir)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))
    items = [(p, i) for i, c in enumerate(classes) for p in sorted((root / c).iterdir())
             if p.suffix.lower() in _IMG_EXTS]
    if not items:
        raise FileNotFoundError(f"no images under {root}/<class>/")

    rng = np.random.default_rng(0)

    class Cards(Dataset):
        def __init__(self, items, augment):
            self.items, self.augment = items, augment

        def __len__(self):
            return len(self.items)

        def __getitem__(self, k):
            p, y = self.items[k]
            img = cv2.imread(str(p))
            if self.augment:
                if rng.random() < 0.3:  # simulate the greyed-out "can't afford" state
                    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.cvtColor((g * 0.6).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
                h, w = img.shape[:2]
                dx, dy = (int(rng.integers(0, max(1, w // 12))) for _ in range(2))
                img = img[dy:h - dy or h, dx:w - dx or w]
            return _to_tensor([img])[0], y

    n_val = max(1, int(len(items) * val_frac)) if len(items) > 4 else 0
    idx = list(range(len(items)))
    train_idx, val_idx = random_split(idx, [len(items) - n_val, n_val], torch.Generator().manual_seed(0))
    train_dl = DataLoader(Cards([items[i] for i in train_idx], True), batch, shuffle=True)
    val_dl = DataLoader(Cards([items[i] for i in val_idx], False), batch)

    model = build_resnet(len(classes), pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    acc = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
                total += len(y)
        acc = correct / max(total, 1)
        print(f"epoch {ep + 1}/{epochs}  loss {loss.item():.4f}  val_acc {acc:.3f}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes}, out)
    return acc
