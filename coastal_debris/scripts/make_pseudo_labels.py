"""비라벨 0.3m 해안 패치에 대한 의사 라벨 생성 (predict_folder.py 의 npz 사용).

  python scripts/predict_folder.py --ckpt <교사모델들> --images data/unlabeled/images --out runs/unl.npz
  python scripts/make_pseudo_labels.py --probs runs/unl.npz --images data/unlabeled/images --out data/pseudo

- 확률 >= --hi : 쓰레기(1), <= --lo : 배경(0), 그 사이 : 무시(ignore/ 에 기록)
- 패치 최대확률이 --neg-max 미만이면 확실한 음성 패치로 보고 전부 배경으로 둡니다.
- 교사의 오류가 그대로 학습되므로, 한 라운드마다 검증셋 점수로 효과를 확인하십시오.
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
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--neg-max", type=float, default=0.3)
    ap.add_argument("--min-pos-px", type=int, default=30, help="확실 양성 화소가 이보다 적은 애매한 패치는 버림")
    a = ap.parse_args()

    z = np.load(a.probs)
    out = Path(a.out)
    for d in ("images", "masks", "ignore"):
        (out / d).mkdir(parents=True, exist_ok=True)
    kept = {"pos": 0, "neg": 0, "skip": 0}
    for i, p in zip(z["ids"], z["seg"].astype(np.float32)):
        name = f"{i}.png"
        if p.max() < a.neg_max:
            m, ign, kind = np.zeros_like(p, np.uint8), None, "neg"
        else:
            m = (p >= a.hi).astype(np.uint8)
            if m.sum() < a.min_pos_px:
                kept["skip"] += 1
                continue
            ign, kind = ((p > a.lo) & (p < a.hi)).astype(np.uint8), "pos"
        shutil.copy(Path(a.images) / name, out / "images" / name)
        cv2.imwrite(str(out / "masks" / name), m * 255)
        if ign is not None and ign.any():
            cv2.imwrite(str(out / "ignore" / name), ign * 255)
        kept[kind] += 1
    print(kept)


if __name__ == "__main__":
    main()
