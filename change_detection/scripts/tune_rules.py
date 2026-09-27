"""검증셋 확률 지도 + 정답 마스크로 클래스별 판정·후처리 규칙을 '대회 산식 그대로' 격자 탐색합니다.
최종 점수 = 0.5*증축 + 0.5*벌목 이고 두 클래스는 서로 영향을 주지 않으므로 클래스별로 따로 탐색합니다.

  python scripts/tune_rules.py --probs runs/val.npz --val data/val --out runs/rules.json
  python scripts/tune_rules.py ... --pos-ratio new_building=0.3 tree_removal=0.2   # 양성 비율 재가중

형상 점수는 래스터 근사(3x3 팽창)로 계산합니다. 최종 확인은 validate_submission.py --val 로
폴리곤 정확 계산을 하십시오. 상위 설정들 차이가 표준오차보다 작으면 덜 극단적인 설정을 고르십시오.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from cdkit import CLASSES, MIN_POS_AREA
from cdkit.data import read_mask
from cdkit.metric import class_score, prevalence_weights, shape_f_mask
from cdkit.postprocess import BASELINE_RULES, DEFAULT_CLASS_RULES, DEFAULT_RULES, make_mask

MASK_GRID = {
    "thr_abs": [0.3, 0.4, 0.5, 0.6],
    "thr_rel": [1.0, 0.7],
    "close": [0, 3, 5],
    "min_cc": [0, 15, 30, 60],
    "guarantee_px": [0, 24, 40],
}
T_MAX = [float(round(x, 2)) for x in np.arange(0.2, 0.96, 0.05)]
MIN_AREA = [20, 30, 50, 80, 120, 200]
T_CLS = [None] + [float(round(x, 2)) for x in np.arange(0.1, 0.96, 0.05)]
FILES = {"new_building": "building.png", "tree_removal": "tree.png"}


def load_gt(val_dir, ids, cls):
    out = []
    for i in ids:
        p = Path(val_dir) / i / FILES[cls]
        out.append(read_mask(p) if p.exists() else np.zeros((256, 256), np.uint8))
    return out


def mask_stats(prob, valid, gts, r):
    area, final, shape = [], [], []
    for p, v, g in zip(prob, valid, gts):
        m, a = make_mask(p.astype(np.float32), r, v)
        area.append(a)
        final.append(int(m.sum()))
        shape.append(shape_f_mask(m, g) if g.any() else 0.0)
    return np.array(area), np.array(final), np.array(shape)


def score(maxp, area, final, shape, cls, gt_pos, g, w):
    pos = (maxp >= g["t_max"]) & (area >= g["min_area"])
    if g.get("t_cls") is not None and cls is not None:
        pos |= cls >= g["t_cls"]
    pos &= final >= MIN_POS_AREA
    return class_score(gt_pos, pos, shape, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--val", required=True, help="<id>/building.png, tree.png 가 있는 검증 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-ratio", nargs="*", default=[], help="class=비율")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    ratios = {k: float(v) for k, v in (s.split("=") for s in a.pos_ratio)}

    z = np.load(a.probs)
    ids, seg, valid = list(z["ids"]), z["seg"], z["valid"]
    cls_all = z["cls"].astype(np.float32)
    best_rules, summary = {}, {}
    for k, c in enumerate(CLASSES):
        prob = seg[:, k]
        cls = None if np.isnan(cls_all[:, k]).all() else cls_all[:, k]
        gts = load_gt(a.val, ids, c)
        gt_pos = np.array([g.any() for g in gts])
        w = prevalence_weights(gt_pos, ratios[c]) if c in ratios else None
        npos = int(gt_pos.sum())
        print(f"\n=== {c}: 검증 쌍 {len(ids)} (양성 {npos}), 분류헤드 {'있음' if cls is not None else '없음'}")
        maxp = prob.reshape(len(prob), -1).max(1).astype(np.float32)
        ref = {}
        for name, r in (("baseline", BASELINE_RULES[c]), ("default", DEFAULT_RULES[c])):
            area, final, shape = mask_stats(prob, valid, gts, r)
            ref[name] = score(maxp, area, final, shape, cls, gt_pos, r, w)
            print(f"[{name}] {ref[name]['score']:.4f} f1 {ref[name]['f1']:.3f} shape {ref[name]['shape']:.3f}")
        results = []
        combos = list(itertools.product(*MASK_GRID.values()))
        for ci, vals in enumerate(combos):
            mr = {**DEFAULT_CLASS_RULES, **dict(zip(MASK_GRID, vals))}
            area, final, shape = mask_stats(prob, valid, gts, mr)
            for t_max, min_area, t_cls in itertools.product(T_MAX, MIN_AREA, T_CLS if cls is not None else [None]):
                g = {**mr, "t_max": t_max, "min_area": min_area, "t_cls": t_cls}
                results.append((score(maxp, area, final, shape, cls, gt_pos, g, w), g))
            if ci % 50 == 0:
                print(f"  mask 규칙 {ci + 1}/{len(combos)}", flush=True)
        results.sort(key=lambda t: -t[0]["score"])
        se = 0.0 if npos == 0 else 0.5 * 0.35 / np.sqrt(npos)
        print(f"상위 {a.top}개 (형상 점수 기여분 표준오차 ≈ ±{se:.3f})")
        for r, g in results[:a.top]:
            print(f"  {r['score']:.4f} f1 {r['f1']:.3f} shape {r['shape']:.3f} fp {r['fp']:.0f} fn {r['fn']:.0f} {g}")
        best_rules[c] = results[0][1]
        summary[c] = (ref["baseline"]["score"], results[0][0]["score"])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(best_rules, indent=2, ensure_ascii=False))
    b = 0.5 * sum(v[0] for v in summary.values())
    t = 0.5 * sum(v[1] for v in summary.values())
    print(f"\n최종(근사): baseline {b:.4f} -> tuned {t:.4f} ({t - b:+.4f})  -> {a.out}")


if __name__ == "__main__":
    main()
