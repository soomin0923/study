"""파이프라인 점검용 가짜 해안 패치 생성 (성능 평가용 아님).
바다(어두운 남색) / 모래(황갈색) + 물가 포말(흰 선, 음성) + 얼룩덜룩한 쓰레기 띠(양성).

  python tests/make_synthetic.py --out /tmp/syn --n 40
  -> <out>/{train,val}/{images,masks}, <out>/eval/{patches.csv,images}
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def patch(rng, positive):
    img = np.zeros((256, 256, 3), np.float32)
    shore = int(rng.integers(60, 200))
    img[:, :shore] = [20, 30, 55]
    img[:, shore:] = rng.uniform([170, 150, 110], [220, 200, 160])
    img[:, shore - 3:shore + 2] = 235                             # 포말 (정답 제외 대상)
    img += rng.normal(0, 6, img.shape)
    m = np.zeros((256, 256), np.uint8)
    if positive:
        y0 = int(rng.integers(0, 150))
        band = np.zeros((256, 256), np.uint8)
        band[y0:y0 + int(rng.integers(20, 100)), shore + 3:shore + int(rng.integers(8, 30))] = 1
        speck = (rng.random((256, 256)) < 0.6) & (band > 0)
        colors = np.array([[240, 240, 240], [150, 150, 150], [60, 90, 200], [70, 160, 80]], np.float32)
        img[speck] = colors[rng.integers(0, 4, speck.sum())]
        m = cv2.morphologyEx(speck.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return np.clip(img, 0, 255).astype(np.uint8), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    for split in ("train", "val"):
        for d in ("images", "masks"):
            (out / split / d).mkdir(parents=True, exist_ok=True)
        for k in range(a.n):
            img, m = patch(rng, k % 2 == 0)
            cv2.imwrite(str(out / split / "images" / f"{split}{k:04d}.png"), img[..., ::-1])
            cv2.imwrite(str(out / split / "masks" / f"{split}{k:04d}.png"), m * 255)
    (out / "eval" / "images").mkdir(parents=True, exist_ok=True)
    with (out / "eval" / "patches.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for k in range(a.n):
            img, _ = patch(rng, k % 3 == 0)
            i = f"m{k + 1:04d}"
            cv2.imwrite(str(out / "eval" / "images" / f"{i}.png"), img[..., ::-1])
            w.writerow([i])
    print("done", out)


if __name__ == "__main__":
    main()
