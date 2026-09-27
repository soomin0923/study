"""대회 평가 산식의 로컬 구현.

클래스 점수 = 0.5 * macroF1(쌍 단위 존재 판정) + 0.5 * 형상 점수
최종 점수   = 0.5 * 증축 점수 + 0.5 * 벌목 점수
- 정답: 그 클래스 폴리곤이 있으면 양성
- 예측: 폴리곤 합집합 면적(패치 밖 잘라낸 뒤) >= 20 이면 양성, 미만이면 두 층 모두 빈 예측
- 형상: 정답 양성 쌍마다 |P ∩ dil1(G)|/|P| 와 |G ∩ dil1(P)|/|G| 의 조화평균, 음성 예측이면 0
  dil1 = 경계를 바깥으로 1화소 (모서리 뾰족하게 연장 = mitre)

두 가지 구현을 둡니다.
- 폴리곤 정확 계산 (shape_f_poly, score_submission) : 제출 CSV 검증용
- 래스터 근사 (shape_f_mask) : 규칙 격자 탐색용. 축 정렬 도형에서는 3x3 팽창과 같아 거의 일치합니다
두 클래스가 점수에서 분리되어 있으므로 규칙 탐색도 클래스별로 따로 합니다.
"""
import cv2
import numpy as np

from . import CLASSES, MIN_POS_AREA

_K = np.ones((3, 3), np.uint8)


def dil1_geom(g):
    return g.buffer(1.0, join_style="mitre", mitre_limit=10.0)


def shape_f_poly(pred_g, gt_g) -> float:
    pa, ga = pred_g.area, gt_g.area
    if pa <= 0 or ga <= 0:
        return 0.0
    prec = pred_g.intersection(dil1_geom(gt_g)).area / pa
    rec = gt_g.intersection(dil1_geom(pred_g)).area / ga
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def shape_f_mask(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 or gs == 0:
        return 0.0
    prec = (p & (cv2.dilate(g.astype(np.uint8), _K) > 0)).sum() / ps
    rec = (g & (cv2.dilate(p.astype(np.uint8), _K) > 0)).sum() / gs
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def class_score(gt_pos, pred_pos, shape_f, weights=None) -> dict:
    gt_pos, pred_pos = np.asarray(gt_pos, bool), np.asarray(pred_pos, bool)
    shape_f = np.asarray(shape_f, float)
    w = np.ones(len(gt_pos)) if weights is None else np.asarray(weights, float)
    tp, fn = w[gt_pos & pred_pos].sum(), w[gt_pos & ~pred_pos].sum()
    fp, tn = w[~gt_pos & pred_pos].sum(), w[~gt_pos & ~pred_pos].sum()
    f1 = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
    wp = w[gt_pos].sum()
    shape = 0.0 if wp == 0 else float((w[gt_pos] * np.where(pred_pos[gt_pos], shape_f[gt_pos], 0)).sum() / wp)
    return {"score": 0.5 * f1 + 0.5 * shape, "f1": f1, "shape": shape,
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def class_score_masks(preds, gts, weights=None) -> dict:
    """마스크 기반 (래스터 근사). 예측 면적 = 화소 수."""
    gp, pp, sf = [], [], []
    for p, g in zip(preds, gts):
        pos, gpos = int(np.count_nonzero(p)) >= MIN_POS_AREA, bool(np.any(g))
        gp.append(gpos)
        pp.append(pos)
        sf.append(shape_f_mask(p, g) if (pos and gpos) else 0.0)
    return class_score(gp, pp, sf, weights)


def score_geoms(pred_geoms: dict, gt_geoms: dict, ids) -> dict:
    """{class: {id: shapely geometry}} 로 최종 점수 (정확 계산)."""
    out = {}
    for c in CLASSES:
        gp, pp, sf = [], [], []
        for i in ids:
            pg, gg = pred_geoms[c].get(i), gt_geoms[c].get(i)
            pos = pg is not None and pg.area >= MIN_POS_AREA
            gpos = gg is not None and gg.area > 0
            gp.append(gpos)
            pp.append(pos)
            sf.append(shape_f_poly(pg, gg) if (pos and gpos) else 0.0)
        out[c] = class_score(gp, pp, sf)
    out["score"] = 0.5 * out[CLASSES[0]]["score"] + 0.5 * out[CLASSES[1]]["score"]
    return out


def prevalence_weights(gt_pos, target_pos_ratio: float) -> np.ndarray:
    gt_pos = np.asarray(gt_pos, bool)
    n, npos = len(gt_pos), gt_pos.sum()
    if npos == 0 or npos == n:
        return np.ones(n)
    return np.where(gt_pos, target_pos_ratio * n / npos, (1 - target_pos_ratio) * n / (n - npos))


def mean_neg_f1_from_all_empty(score_all_empty: float) -> float:
    """전부 빈 예측 제출 점수 -> 두 클래스 '음성 F1' 의 평균.
    s0 = 0.5*(0.25*Nb) + 0.5*(0.25*Nt) = 0.125*(Nb+Nt),  N = 2(1-p)/(2-p)
    두 클래스 양성 비율이 비슷하다고 가정하면 p ≈ (2-2N)/(2-N)."""
    n = 4 * score_all_empty
    return n, (2 - 2 * n) / (2 - n)
