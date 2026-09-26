"""학습용 데이터셋과 augmentation (numpy/cv2 만 사용).

데이터 소스 폴더 규약 (모든 소스 공통):
  <src>/images/<id>.png   256x256 RGB
  <src>/masks/<id>.png    0 = 배경, >0 = 쓰레기   (폴더가 없으면 전부 배경 = 음성 전용 소스)
  <src>/ignore/<id>.png   >0 = 학습에서 무시할 화소 (의사 라벨의 불확실 영역, 선택)
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import IGNORE


def read_rgb(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_mask(path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def list_source(src) -> list[dict]:
    src = Path(src)
    items = []
    for p in sorted((src / "images").glob("*.png")):
        mp, ip = src / "masks" / p.name, src / "ignore" / p.name
        items.append({"id": p.stem, "image": p, "mask": mp if mp.exists() else None,
                      "ignore": ip if ip.exists() else None})
    if not items:
        raise FileNotFoundError(f"{src}/images/*.png 가 없습니다")
    return items


def load_item(it) -> tuple[np.ndarray, np.ndarray]:
    img = read_rgb(it["image"])
    tgt = read_mask(it["mask"]) if it["mask"] else np.zeros(img.shape[:2], np.uint8)
    if it["ignore"]:
        tgt[read_mask(it["ignore"]) > 0] = IGNORE
    return img, tgt


# ---------------- augmentation ----------------

def _d4(img, tgt, rng):
    k = int(rng.integers(8))
    if k >= 4:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    return np.rot90(img, k % 4).copy(), np.rot90(tgt, k % 4).copy()


def _scale(img, tgt, rng, lo=0.85, hi=1.15):
    s = rng.uniform(lo, hi)
    h, w = tgt.shape
    nh, nw = int(round(h * s)), int(round(w * s))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    tgt = cv2.resize(tgt, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if s >= 1:
        y, x = rng.integers(nh - h + 1), rng.integers(nw - w + 1)
        return img[y:y + h, x:x + w], tgt[y:y + h, x:x + w]
    py, px = h - nh, w - nw
    t, l = rng.integers(py + 1), rng.integers(px + 1)
    img = cv2.copyMakeBorder(img, t, py - t, l, px - l, cv2.BORDER_REFLECT_101)
    tgt = cv2.copyMakeBorder(tgt, t, py - t, l, px - l, cv2.BORDER_REFLECT_101)
    return img, tgt


def color_aug(img, rng, strength=1.0):
    """장면별 톤 보정 차이(누렇고 뿌연 영상 ~ 채도 높은 영상)를 흉내냅니다."""
    x = np.ascontiguousarray(img).astype(np.float32)
    hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8, 8) * strength) % 180
    hsv[..., 1] *= 1 + rng.uniform(-0.4, 0.5) * strength
    hsv[..., 2] *= 1 + rng.uniform(-0.25, 0.25) * strength
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    x *= 1 + rng.uniform(-0.12, 0.12, size=3) * strength             # 채널별 gain
    mean = x.mean()
    x = (x - mean) * (1 + rng.uniform(-0.3, 0.3) * strength) + mean  # 대비
    g = np.exp(rng.uniform(-0.35, 0.35) * strength)                  # 감마
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** g
    if rng.random() < 0.3 * strength:                                # 뿌연 막(haze), 누런 톤 포함
        a = rng.uniform(0.05, 0.35)
        haze = np.array([rng.uniform(170, 240), rng.uniform(160, 230), rng.uniform(120, 220)], np.float32)
        x = x * (1 - a) + haze * a
    return np.clip(x, 0, 255).astype(np.uint8)


def degrade_aug(img, rng):
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, rng.uniform(1.5, 6), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        _, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 96))])
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img)


def augment(img, tgt, rng, color_strength=1.0):
    img, tgt = _d4(img, tgt, rng)
    if rng.random() < 0.5:
        img, tgt = _scale(img, tgt, rng)
    img = color_aug(img, rng, color_strength)
    img = degrade_aug(img, rng)
    return img, tgt


class DebrisDataset(Dataset):
    def __init__(self, items, train=True, color_strength=1.0, seed=0):
        self.items, self.train, self.cs, self.seed = items, train, color_strength, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, tgt = load_item(self.items[i])
        if self.train:
            rng = np.random.default_rng((self.seed, i, int(torch.randint(1 << 30, (1,)))))
            img, tgt = augment(img, tgt, rng, self.cs)
        from .model import normalize
        x = normalize(img[None])[0]
        return x, torch.from_numpy(tgt.astype(np.int64))
