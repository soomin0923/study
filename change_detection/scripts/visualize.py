"""오류 분석용 시각화. 클래스별 FP / FN / 형상 점수 낮은 쌍을 PNG 로 모읍니다.

  python scripts/visualize.py --probs runs/val.npz --val data/val --rules runs/rules.json --out runs/vis

각 줄: [전 영상 | 후 영상 | 정답(초록) | 예측(빨강) | 확률 지도]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT  # noqa: F401
from cdkit import CLASSES, MIN_POS_AREA
from cdkit.data import read_mask, read_rgb
from cdkit.metric import shape_f_mask
from cdkit.postprocess import DEFAULT_CLASS_RULES, DEFAULT_RULES, argmax_filter, decide_class

FILES = {"new_building": "building.png", "tree_removal": "tree.png"}


def overlay(img, m, color):
    o = img.copy()
    o[m > 0] = (0.45 * o[m > 0] + 0.55 * np.array(color)).astype(np.uint8)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--rules", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-page", type=int, default=10)
    a = ap.parse_args()
    rules = json.loads(Path(a.rules).read_text()) if a.rules else DEFAULT_RULES
    z = np.load(a.probs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate(CLASSES):
        r = {**DEFAULT_CLASS_RULES, **rules.get(c, {})}
        groups = {"FP": [], "FN": [], "low_shape": []}
        for n, i in enumerate(z["ids"]):
            seg_n = z["seg"][n].astype(np.float32)
            p = argmax_filter(seg_n, k) if r.get("argmax") else seg_n[k]
            cp = None if np.isnan(z["cls"][n, k]) else float(z["cls"][n, k])
            gp = Path(a.val) / i / FILES[c]
            g = read_mask(gp) if gp.exists() else np.zeros((256, 256), np.uint8)
            m = decide_class(p, cp, r, z["valid"][n])
            pos, gpos = m.sum() >= MIN_POS_AREA, g.any()
            cap = f"{i} {c} max{p.max():.2f}" + ("" if cp is None else f" cls{cp:.2f}")
            if pos and not gpos:
                groups["FP"].append((-p.max(), i, g, m, p, cap))
            elif gpos and not pos:
                groups["FN"].append((p.max(), i, g, m, p, cap + f" gt{int(g.sum())}px"))
            elif gpos and pos:
                f = shape_f_mask(m, g)
                groups["low_shape"].append((f, i, g, m, p, cap + f" F{f:.2f}"))
        for name, lst in groups.items():
            lst.sort(key=lambda t: t[0])
            for pg in range(0, len(lst), a.per_page):
                rows = []
                for _, i, g, m, p, cap in lst[pg:pg + a.per_page]:
                    pre, post = read_rgb(Path(a.val) / i / "pre.png"), read_rgb(Path(a.val) / i / "post.png")
                    heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
                    row = np.ascontiguousarray(np.concatenate(
                        [pre, post, overlay(post, g, (0, 255, 0)), overlay(post, m, (255, 0, 0)), heat], 1))
                    cv2.putText(row, cap, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
                    rows.append(row)
                cv2.imwrite(str(out / f"{c}_{name}_{pg // a.per_page:02d}.png"), np.concatenate(rows, 0)[..., ::-1])
            print(f"{c} {name}: {len(lst)}개")


if __name__ == "__main__":
    main()
