"""학습용 데이터셋과 augmentation (numpy/cv2 만 사용).

데이터 소스 폴더 규약:
  <src>/<id>/pre.png        전 영상 256x256 RGB
  <src>/<id>/post.png       후 영상 (같은 격자)
  <src>/<id>/building.png   증축 마스크 (>0 = 증축)   없으면 전부 0
  <src>/<id>/tree.png       벌목 마스크 (>0 = 벌목)   없으면 전부 0
  <src>/<id>/ignore.png     학습에서 무시할 화소 (선택, 의사 라벨의 불확실 영역)

augmentation 은 이 과제의 '변화가 아닌 차이'를 일부러 만들어 모델이 속지 않게 합니다.
- 전/후 영상에 서로 다른 밝기·색 조정 (영상별 따로 밝기를 편 규격)
- 계절 변화 흉내: 한쪽 영상의 녹색을 갈색 쪽으로 (잎 진 산림, 작물 상태)
- 시차 흉내: 전 영상만 수 화소 이동 (정답은 후 영상 격자 기준이므로 후 영상은 고정)
- 무영상 흉내: 한쪽 영상 가장자리를 직선 경계로 검게, 그 구역은 학습에서 무시
- 같은 영상 쌍: 후 영상을 두 번 쓰고 색만 다르게 -> 정답은 '변화 없음'
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .losses import IGNORE


def read_rgb(path) -> np.ndarray:
    im = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def read_mask(path) -> np.ndarray:
    m = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def list_source(src) -> list[dict]:
    src = Path(src)
    items = []
    for d in sorted(p for p in src.iterdir() if (p / "pre.png").exists() and (p / "post.png").exists()):
        def opt(name):
            return d / name if (d / name).exists() else None
        items.append({"id": d.name, "pre": d / "pre.png", "post": d / "post.png",
                      "building": opt("building.png"), "tree": opt("tree.png"), "ignore": opt("ignore.png")})
    if not items:
        raise FileNotFoundError(f"{src}/<id>/pre.png, post.png 가 없습니다")
    return items


def load_target(it) -> np.ndarray:
    """0 배경 / 1 증축 / 2 벌목 / 255 무시. 두 클래스가 겹치면 증축 우선."""
    shape = (256, 256)
    t = np.zeros(shape, np.uint8)
    if it.get("tree"):
        t[read_mask(it["tree"]) > 0] = 2
    if it.get("building"):
        t[read_mask(it["building"]) > 0] = 1
    if it.get("ignore"):
        t[read_mask(it["ignore"]) > 0] = IGNORE
    return t


def load_item(it):
    return read_rgb(it["pre"]), read_rgb(it["post"]), load_target(it)


# ---------------- augmentation ----------------

def color_aug(img, rng, s=1.0):
    x = np.ascontiguousarray(img)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-6, 6) * s) % 180
    hsv[..., 1] *= 1 + rng.uniform(-0.35, 0.4) * s
    hsv[..., 2] *= 1 + rng.uniform(-0.25, 0.25) * s
    x = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    x *= 1 + rng.uniform(-0.12, 0.12, size=3) * s
    mean = x.mean()
    x = (x - mean) * (1 + rng.uniform(-0.3, 0.3) * s) + mean
    x = 255.0 * (np.clip(x, 0, 255) / 255.0) ** np.exp(rng.uniform(-0.3, 0.3) * s)
    if rng.random() < 0.25 * s:
        a = rng.uniform(0.05, 0.3)
        x = x * (1 - a) + np.array([rng.uniform(170, 235)] * 3, np.float32) * a
    return np.clip(x, 0, 255).astype(np.uint8)


def season_aug(img, rng):
    """녹색 화소의 색상을 황갈색 쪽으로 옮기고 채도를 낮춤 (잎 진 산림, 작물 변화 흉내)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
    green = (hsv[..., 0] > 30) & (hsv[..., 0] < 90)
    hsv[..., 0][green] -= rng.uniform(12, 25)
    hsv[..., 1][green] *= rng.uniform(0.5, 0.9)
    hsv[..., 0] %= 180
    return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def shift(img, dx, dy):
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)


def nodata_halfplane(rng, shape=(256, 256)):
    """가장자리에 직선 경계로 들어오는 무영상 영역."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    ang = rng.uniform(0, 2 * np.pi)
    nx, ny = np.cos(ang), np.sin(ang)
    proj = (xx - w / 2) * nx + (yy - h / 2) * ny
    lim = np.abs(proj).max()
    return proj > rng.uniform(0.35, 0.9) * lim


def augment(pre, post, tgt, rng, cs=1.0, p_same=0.1, p_season=0.3, p_shift=0.5, p_nodata=0.15):
    # 같은 영상 쌍 -> 변화 없음
    if rng.random() < p_same:
        pre = post.copy()
        tgt = np.where(tgt == IGNORE, IGNORE, 0).astype(np.uint8)
    # 공통 기하 변환
    k = int(rng.integers(8))
    if k >= 4:
        pre, post, tgt = pre[:, ::-1], post[:, ::-1], tgt[:, ::-1]
    pre, post, tgt = (np.ascontiguousarray(np.rot90(a, k % 4)) for a in (pre, post, tgt))
    # 영상별 색 조정
    pre, post = color_aug(pre, rng, cs), color_aug(post, rng, cs)
    if rng.random() < p_season:
        if rng.random() < 0.5:
            pre = season_aug(pre, rng)
        else:
            post = season_aug(post, rng)
    # 시차: 전 영상만 이동
    if rng.random() < p_shift:
        pre = shift(pre, *rng.uniform(-3, 3, size=2))
    # 무영상
    if rng.random() < p_nodata:
        nd = nodata_halfplane(rng)
        (pre if rng.random() < 0.5 else post)[nd] = 0
        tgt = tgt.copy()
        tgt[nd] = IGNORE
    # 약한 블러/노이즈
    if rng.random() < 0.25:
        post = cv2.GaussianBlur(post, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.25:
        pre = np.clip(pre + rng.normal(0, rng.uniform(1, 5), pre.shape), 0, 255).astype(np.uint8)
    return pre, post, tgt


class CDDataset(Dataset):
    def __init__(self, items, train=True, color_strength=1.0, p_same=0.1, seed=0):
        self.items, self.train, self.cs, self.p_same, self.seed = items, train, color_strength, p_same, seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from .model import normalize_pair
        pre, post, tgt = load_item(self.items[i])
        if self.train:
            rng = np.random.default_rng((self.seed, i, int(torch.randint(1 << 30, (1,)))))
            pre, post, tgt = augment(pre, post, tgt, rng, self.cs, self.p_same)
        x = normalize_pair(pre[None], post[None])[0]
        return x, torch.from_numpy(tgt.astype(np.int64))
