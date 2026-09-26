"""AIHub 국립공원 변화탐지 데이터(드론 0.1m) -> 0.3m 위성 흉내 256x256 학습 패치.

입력: 드론 원본 영상 폴더 + 같은 파일명(확장자 무관)의 클래스 마스크 PNG 폴더.
  AIHub 라벨이 폴리곤 JSON 이라면, 먼저 클래스 마스크 PNG 로 래스터화한 뒤 이 스크립트를 쓰십시오.
  (데이터를 받은 뒤 라벨 형식과 '해안쓰레기' 클래스 값을 확인해 --debris-values 로 넘깁니다.)

  python scripts/prepare_aihub.py --images aihub/drone/images --masks aihub/drone/labels \
      --debris-values 7 --out data/aihub_sat --neg-out data/aihub_neg

- 0.1m 에서 768x768 창으로 자르고(0.3m 256x256 에 해당) degrade.py 로 위성처럼 만듭니다.
- 쓰레기가 있는 창은 --out, 없는 창은 --neg-keep 확률로 --neg-out 에 저장합니다(해안 hard negative).
- 이 데이터를 쓸지와 비율은 반드시 0.3m 검증셋 점수로 결정하십시오. AIHub 이용약관도 확인하십시오.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT  # noqa: F401
from debris_kit.degrade import degrade_image, degrade_mask

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--debris-values", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-out", default=None)
    ap.add_argument("--neg-keep", type=float, default=0.3)
    ap.add_argument("--src-gsd", type=float, default=0.1)
    ap.add_argument("--dst-gsd", type=float, default=0.3)
    ap.add_argument("--stride-frac", type=float, default=0.75, help="창 이동 간격 / 창 크기")
    ap.add_argument("--frac-thr", type=float, default=0.4, help="축소 마스크 임계 (화소 면적 비율)")
    ap.add_argument("--close", type=int, default=0, help="개체 라벨을 집적대 단위로 잇는 closing 크기 (0.3m 화소)")
    ap.add_argument("--min-pos-px", type=int, default=20, help="축소 후 이보다 작으면 양성 창으로 보지 않음")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    factor = a.dst_gsd / a.src_gsd
    win = int(round(256 * factor))
    stride = max(1, int(win * a.stride_frac))
    mask_by_stem = {p.stem: p for p in Path(a.masks).rglob("*") if p.suffix.lower() in EXTS}
    outs = {"pos": Path(a.out), "neg": Path(a.neg_out) if a.neg_out else None}
    for o in outs.values():
        if o:
            (o / "images").mkdir(parents=True, exist_ok=True)
    (outs["pos"] / "masks").mkdir(parents=True, exist_ok=True)

    n_pos = n_neg = 0
    for ip in sorted(p for p in Path(a.images).rglob("*") if p.suffix.lower() in EXTS):
        mp = mask_by_stem.get(ip.stem)
        if mp is None:
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        lab = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if lab.ndim == 3:
            lab = lab[..., 0]
        deb = np.isin(lab, a.debris_values)
        h, w = deb.shape
        for y in range(0, max(1, h - win + 1), stride):
            for x in range(0, max(1, w - win + 1), stride):
                ti, tm = img[y:y + win, x:x + win], deb[y:y + win, x:x + win]
                if ti.shape[:2] != (win, win):
                    continue
                sm = degrade_mask(tm, factor, a.frac_thr, a.close)
                si = degrade_image(ti, factor, rng)
                if si.shape[:2] != (256, 256):
                    si = cv2.resize(si, (256, 256), interpolation=cv2.INTER_AREA)
                    sm = cv2.resize(sm, (256, 256), interpolation=cv2.INTER_NEAREST)
                name = f"{ip.stem}_{y}_{x}.png"
                if sm.sum() >= a.min_pos_px:
                    cv2.imwrite(str(outs["pos"] / "images" / name), si[..., ::-1])
                    cv2.imwrite(str(outs["pos"] / "masks" / name), sm * 255)
                    n_pos += 1
                elif tm.sum() == 0 and outs["neg"] and rng.random() < a.neg_keep:
                    cv2.imwrite(str(outs["neg"] / "images" / name), si[..., ::-1])
                    n_neg += 1
        print(f"{ip.name}: 누적 양성 {n_pos}, 음성 {n_neg}", flush=True)
    print("done", n_pos, n_neg)


if __name__ == "__main__":
    main()
