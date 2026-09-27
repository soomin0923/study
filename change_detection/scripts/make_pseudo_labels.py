"""라벨 없는 전/후 쌍에 대한 의사 라벨 (predict_folder.py 의 npz 사용).

  python scripts/predict_folder.py --ckpt <교사모델들> --pairs data/unlabeled --out runs/unl.npz
  python scripts/make_pseudo_labels.py --probs runs/unl.npz --pairs data/unlabeled --out data/pseudo

- 클래스별로 확률 >= --hi 는 정답, 두 클래스 모두 <= --lo 는 배경, 그 사이는 무시(ignore.png)
- 두 클래스 모두 최대확률이 --neg-max 미만이면 '변화 없음' 쌍으로 전부 배경
- 교사 오류가 그대로 학습되므로 라운드마다 검증셋 점수로 확인하십시오
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--neg-max", type=float, default=0.3)
    ap.add_argument("--min-pos-px", type=int, default=30)
    a = ap.parse_args()
    z = np.load(a.probs)
    stat = {"pos": 0, "neg": 0, "skip": 0}
    for n, i in enumerate(z["ids"]):
        p = z["seg"][n].astype(np.float32)          # (2,H,W)
        valid = z["valid"][n]
        if p.max() < a.neg_max:
            masks, ign, kind = [np.zeros(p.shape[1:], np.uint8)] * 2, ~valid, "neg"
        else:
            masks = [(q >= a.hi).astype(np.uint8) for q in p]
            if max(int(m.sum()) for m in masks) < a.min_pos_px:
                stat["skip"] += 1
                continue
            ign, kind = (((p > a.lo) & (p < a.hi)).any(0)) | ~valid, "pos"
        d = Path(a.out) / i
        d.mkdir(parents=True, exist_ok=True)
        for f in ("pre.png", "post.png"):
            shutil.copy(Path(a.pairs) / i / f, d / f)
        cv2.imwrite(str(d / "building.png"), masks[0] * 255)
        cv2.imwrite(str(d / "tree.png"), masks[1] * 255)
        if ign.any():
            cv2.imwrite(str(d / "ignore.png"), ign.astype(np.uint8) * 255)
        stat[kind] += 1
    print(stat)


if __name__ == "__main__":
    main()
