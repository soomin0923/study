"""검증셋 확률 지도 + 정답 마스크로 판정·후처리 규칙을 '대회 산식 그대로' 격자 탐색합니다.

  python scripts/tune_rules.py --probs runs/val.npz --masks data/val/masks --out runs/rules.json
  # 평가셋 양성 비율을 리더보드 탐색으로 추정했다면 (probe_calc.py) 검증셋을 그 비율로 재가중:
  python scripts/tune_rules.py ... --pos-ratio 0.42

주의: 검증셋이 작으면 격자 최적값이 검증셋에 과적합됩니다. 출력되는 상위 설정들의 점수 차이가
표준오차(출력됨)보다 작으면 '더 단순한/덜 극단적인' 설정을 고르십시오.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401  (sys.path 설정)
from debris_kit import MIN_POS_PX
from debris_kit.data import read_mask
from debris_kit.metric import patch_shape_f, prevalence_weights, score_from_stats
from debris_kit.postprocess import BASELINE_RULES, DEFAULT_RULES, make_mask

MASK_GRID = {
    "thr_abs": [0.3, 0.4, 0.5, 0.6],
    "thr_rel": [1.0, 0.7, 0.5],
    "min_cc": [0, 10, 30],
    "close": [0, 3],
    "guarantee_px": [0, 60, 80],
}
T_MAX = [float(round(x, 2)) for x in np.arange(0.2, 0.96, 0.05)]
MIN_AREA = [0, 10, 25, 50, 100, 200]
T_CLS = [None] + [float(round(x, 2)) for x in np.arange(0.1, 0.96, 0.05)]


def mask_stats(seg, gts, rules):
    """규칙 하나에 대해 패치별 (확장 전 면적, 최종 면적, 형상 F)."""
    area, final, shape = [], [], []
    for p, g in zip(seg, gts):
        m, a = make_mask(p.astype(np.float32), rules)
        area.append(a)
        final.append(int(m.sum()))
        shape.append(patch_shape_f(m, g) if g.any() else 0.0)
    return np.array(area), np.array(final), np.array(shape)


def eval_rules(seg, cls, gts, gt_pos, rules, w=None):
    area, final, shape = mask_stats(seg, gts, rules)
    maxp = seg.reshape(len(seg), -1).max(1).astype(np.float32)
    return _score(maxp, area, final, shape, cls, gt_pos, rules, w)


def _score(maxp, area, final, shape, cls, gt_pos, g, w):
    pos = (maxp >= g["t_max"]) & (area >= g["min_area"])
    if g.get("t_cls") is not None and cls is not None:
        pos |= cls >= g["t_cls"]
    pos &= final >= MIN_POS_PX
    return score_from_stats(gt_pos, pos, shape, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pos-ratio", type=float, default=None)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    z = np.load(a.probs)
    ids, seg = list(z["ids"]), z["seg"]
    cls = z["cls"].astype(np.float32)
    cls = None if np.isnan(cls).all() else cls
    gts = [read_mask(Path(a.masks) / f"{i}.png") for i in ids]
    gt_pos = np.array([g.any() for g in gts])
    w = prevalence_weights(gt_pos, a.pos_ratio) if a.pos_ratio is not None else None
    npos = int(gt_pos.sum())
    print(f"검증 패치 {len(ids)}개 (양성 {npos}, 음성 {len(ids) - npos}), 분류헤드 {'있음' if cls is not None else '없음'}")

    ref = {name: eval_rules(seg, cls, gts, gt_pos, r, w) for name, r in
           [("baseline(0.5 argmax)", BASELINE_RULES), ("default", DEFAULT_RULES)]}
    for k, v in ref.items():
        print(f"[{k}] score {v['score']:.4f}  f1 {v['f1']:.4f}  shape {v['shape']:.4f}")

    maxp = seg.reshape(len(seg), -1).max(1).astype(np.float32)
    keys = list(MASK_GRID)
    results = []
    combos = list(itertools.product(*MASK_GRID.values()))
    for ci, vals in enumerate(combos):
        mr = dict(zip(keys, vals))
        area, final, shape = mask_stats(seg, gts, mr)
        for t_max, min_area, t_cls in itertools.product(T_MAX, MIN_AREA, T_CLS if cls is not None else [None]):
            g = {**mr, "t_max": t_max, "min_area": min_area, "t_cls": t_cls}
            r = _score(maxp, area, final, shape, cls, gt_pos, g, w)
            results.append((r["score"], r, g))
        if ci % 20 == 0:
            print(f"  mask 규칙 {ci + 1}/{len(combos)}", flush=True)
    results.sort(key=lambda t: -t[0])

    # 형상 점수 표준오차의 대략값 (양성 패치 수 기준) -> 상위 설정들의 차이가 의미 있는지 판단용
    best = results[0]
    se = 0.0 if npos == 0 else 0.5 * 0.35 / np.sqrt(npos)  # 패치별 형상 F 표준편차를 0.35 로 가정
    print(f"\n상위 {a.top}개 (형상 점수 기여분 표준오차 ≈ ±{se:.3f})")
    for s, r, g in results[:a.top]:
        print(f"  {s:.4f}  f1 {r['f1']:.3f} shape {r['shape']:.3f}  fp {r['fp']:.0f} fn {r['fn']:.0f}  {g}")

    out = {**DEFAULT_RULES, **best[2]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n최고 {best[0]:.4f} (baseline 대비 {best[0] - ref['baseline(0.5 argmax)']['score']:+.4f}) -> {a.out}")


if __name__ == "__main__":
    main()
