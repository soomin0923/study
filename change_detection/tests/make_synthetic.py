"""파이프라인 점검용 가짜 전/후 쌍 생성 (성능 평가용 아님).
전: 산림(녹색 알갱이) + 밭 + 기존 건물 / 후: 색감 변화 + 새 건물(증축) + 산림 일부 맨땅(벌목) + 가끔 무영상 모서리

  python tests/make_synthetic.py --out /tmp/syn --n 24
  -> <out>/{train,val}/<id>/{pre,post,building,tree}.png, <out>/eval/{pairs.csv,images/<id>/{pre,post}.png}
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def scene(rng):
    img = np.zeros((256, 256, 3), np.float32)
    img[:] = rng.uniform([150, 140, 100], [190, 175, 130])          # 밭/나지
    forest = np.zeros((256, 256), np.uint8)
    for _ in range(3):
        c = tuple(int(v) for v in rng.integers(30, 226, 2))
        cv2.circle(forest, c, int(rng.integers(30, 70)), 1, -1)
    tex = rng.normal(0, 18, (256, 256)).astype(np.float32)
    img[forest > 0] = np.array([40, 90, 45], np.float32) + tex[forest > 0, None]
    old_b = np.zeros((256, 256), np.uint8)
    for _ in range(3):
        y, x = rng.integers(10, 230, 2)
        if forest[y, x] == 0:
            cv2.rectangle(old_b, (int(x), int(y)), (int(x) + 16, int(y) + 14), 1, -1)
    img[old_b > 0] = [200, 200, 205]
    return img, forest


def pair(rng, change):
    pre, forest = scene(rng)
    post = pre * rng.uniform(0.85, 1.15, 3) + rng.uniform(-10, 10)      # 영상별 색감 차이
    bld, tree = np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8)
    if change:
        for _ in range(int(rng.integers(1, 3))):
            y, x = rng.integers(10, 220, 2)
            if forest[y:y + 18, x:x + 22].any():
                continue
            cv2.rectangle(bld, (int(x), int(y)), (int(x) + 21, int(y) + 17), 1, -1)
        post[bld > 0] = rng.choice([[90, 110, 180], [210, 210, 215], [170, 80, 60]])
        if forest.any() and rng.random() < 0.7:
            ys, xs = np.nonzero(forest)
            k = rng.integers(len(ys))
            cv2.circle(tree, (int(xs[k]), int(ys[k])), int(rng.integers(12, 25)), 1, -1)
            tree &= forest
            post[tree > 0] = np.array([175, 140, 95]) + rng.normal(0, 5, (int(tree.sum()), 3))
    if rng.random() < 0.2:                                                # 무영상 모서리
        yy, xx = np.mgrid[0:256, 0:256]
        nd = xx + yy < rng.integers(40, 120)
        post[nd] = 0
        bld[nd] = 0
        tree[nd] = 0
    cl = lambda a: np.clip(a, 0, 255).astype(np.uint8)  # noqa: E731
    return cl(pre), cl(post), bld, tree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = Path(a.out)
    for split in ("train", "val"):
        for k in range(a.n):
            d = out / split / f"{split}{k:04d}"
            d.mkdir(parents=True, exist_ok=True)
            pre, post, b, t = pair(rng, k % 3 != 0)
            cv2.imwrite(str(d / "pre.png"), pre[..., ::-1])
            cv2.imwrite(str(d / "post.png"), post[..., ::-1])
            cv2.imwrite(str(d / "building.png"), b * 255)
            cv2.imwrite(str(d / "tree.png"), t * 255)
    (out / "eval" / "images").mkdir(parents=True, exist_ok=True)
    with (out / "eval" / "pairs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id"])
        for k in range(a.n):
            i = f"u{k + 1:04d}"
            d = out / "eval" / "images" / i
            d.mkdir(parents=True, exist_ok=True)
            pre, post, _, _ = pair(rng, k % 2 == 0)
            cv2.imwrite(str(d / "pre.png"), pre[..., ::-1])
            cv2.imwrite(str(d / "post.png"), post[..., ::-1])
            w.writerow([i])
    print("done", out)


if __name__ == "__main__":
    main()
