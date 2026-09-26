"""오류 분석용 시각화. 검증셋에서 FP / FN / 형상 점수 낮은 패치를 모아 PNG 로 저장합니다.

  python scripts/visualize.py --probs runs/val.npz --images data/val/images --masks data/val/masks \
      --rules runs/rules.json --out runs/vis

각 칸: [원본 | 정답(초록) | 예측(빨강) | 확률 지도]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT  # noqa: F401
from debris_kit import MIN_POS_PX
from debris_kit.data import read_mask, read_rgb
from debris_kit.metric import patch_shape_f
from debris_kit.postprocess import DEFAULT_RULES, decide


def overlay(img, m, color):
    o = img.copy()
    o[m > 0] = (0.45 * o[m > 0] + 0.55 * np.array(color)).astype(np.uint8)
    return o


def tile(img, gt, pred, prob, caption):
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    row = np.concatenate([img, overlay(img, gt, (0, 255, 0)), overlay(img, pred, (255, 0, 0)), heat], 1)
    row = np.ascontiguousarray(row)
    cv2.putText(row, caption, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=12)
    a = ap.parse_args()

    rules = {**DEFAULT_RULES, **(json.loads(Path(a.rules).read_text()) if a.rules else {})}
    z = np.load(a.probs)
    cls = z["cls"]
    groups = {"FP": [], "FN": [], "low_shape": []}
    for k, (i, p) in enumerate(zip(z["ids"], z["seg"].astype(np.float32))):
        c = None if np.isnan(cls[k]) else float(cls[k])
        g = read_mask(Path(a.masks) / f"{i}.png")
        m = decide(p, c, rules)
        pp, gp = m.sum() >= MIN_POS_PX, g.any()
        cap = f"{i} max{p.max():.2f} cls{c if c is None else round(c, 2)}"
        if pp and not gp:
            groups["FP"].append((p.max(), i, g, m, p, cap))
        elif gp and not pp:
            groups["FN"].append((-p.max(), i, g, m, p, cap + f" gt{int(g.sum())}px"))
        elif gp and pp:
            f = patch_shape_f(m, g)
            groups["low_shape"].append((f, i, g, m, p, cap + f" F{f:.2f}"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, lst in groups.items():
        lst.sort(key=lambda t: t[0], reverse=(name == "FP"))
        for pg in range(0, len(lst), a.per_page):
            rows = [tile(read_rgb(Path(a.images) / f"{i}.png"), g, m, p, cap)
                    for _, i, g, m, p, cap in lst[pg:pg + a.per_page]]
            cv2.imwrite(str(out / f"{name}_{pg // a.per_page:02d}.png"), np.concatenate(rows, 0)[..., ::-1])
        print(f"{name}: {len(lst)}개")


if __name__ == "__main__":
    main()
