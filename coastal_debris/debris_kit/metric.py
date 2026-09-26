"""대회 평가 산식의 로컬 구현.

최종 점수 = 0.5 * macroF1(패치 존재 판정) + 0.5 * 형상 점수
- 정답: 1인 화소가 하나라도 있으면 양성
- 예측: 1인 화소가 50개 이상이면 양성 (미만이면 두 층 모두 빈 예측)
- 형상 점수: 정답 양성 패치마다 1화소 허용 F-score, 음성 예측이면 0, 정답 양성 패치 전체 평균

weights 를 주면 패치별 가중치로 계산합니다 (평가셋의 양성 비율에 맞춰 검증셋을 재가중할 때 사용).
"""
import cv2
import numpy as np

from . import MIN_POS_PX

_K = np.ones((3, 3), np.uint8)


def dilate1(m: np.ndarray) -> np.ndarray:
    """상하좌우+대각선 1화소 확장."""
    return cv2.dilate(m.astype(np.uint8), _K).astype(bool)


def patch_shape_f(pred: np.ndarray, gt: np.ndarray) -> float:
    """정답 양성 패치 하나의 형상 점수 (pred 는 이미 양성 예측이라고 가정)."""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = p.sum(), g.sum()
    if ps == 0 or gs == 0:
        return 0.0
    prec = (p & dilate1(g)).sum() / ps
    rec = (g & dilate1(p)).sum() / gs
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def score_from_stats(gt_pos, pred_pos, shape_f, weights=None) -> dict:
    """패치별 (정답 양성, 예측 양성, 형상 F) 배열로 점수를 계산합니다. 규칙 탐색에서 재사용."""
    gt_pos = np.asarray(gt_pos, bool)
    pred_pos = np.asarray(pred_pos, bool)
    shape_f = np.asarray(shape_f, float)
    w = np.ones(len(gt_pos)) if weights is None else np.asarray(weights, float)
    tp = w[gt_pos & pred_pos].sum()
    fn = w[gt_pos & ~pred_pos].sum()
    fp = w[~gt_pos & pred_pos].sum()
    tn = w[~gt_pos & ~pred_pos].sum()
    f1 = (_f1(tp, fp, fn) + _f1(tn, fn, fp)) / 2
    wp = w[gt_pos].sum()
    shape = 0.0 if wp == 0 else float((w[gt_pos] * np.where(pred_pos[gt_pos], shape_f[gt_pos], 0.0)).sum() / wp)
    return {"score": 0.5 * f1 + 0.5 * shape, "f1": f1, "shape": shape,
            "tp": float(tp), "fp": float(fp), "fn": float(fn), "tn": float(tn)}


def competition_score(preds, gts, weights=None) -> dict:
    """preds, gts: (256,256) 0/1 마스크 리스트."""
    gt_pos, pred_pos, shape_f = [], [], []
    for p, g in zip(preds, gts):
        pp = int(np.count_nonzero(p)) >= MIN_POS_PX
        gp = bool(np.any(g))
        gt_pos.append(gp)
        pred_pos.append(pp)
        shape_f.append(patch_shape_f(p, g) if (gp and pp) else 0.0)
    return score_from_stats(gt_pos, pred_pos, shape_f, weights)


def prevalence_weights(gt_pos, target_pos_ratio: float) -> np.ndarray:
    """검증셋의 양성 비율을 target 으로 맞추는 패치 가중치."""
    gt_pos = np.asarray(gt_pos, bool)
    n, npos = len(gt_pos), gt_pos.sum()
    if npos == 0 or npos == n:
        return np.ones(n)
    wp = target_pos_ratio * n / npos
    wn = (1 - target_pos_ratio) * n / (n - npos)
    return np.where(gt_pos, wp, wn)


# ---- 리더보드 탐색 제출(전부 빈 마스크 / 전부 1) 결과로 평가셋 통계를 역산 ----

def pos_ratio_from_all_empty(score_all_empty: float) -> float:
    """전부 빈 마스크 제출 점수 s0 -> 양성 패치 비율 p.
    s0 = 0.25 * F1_neg, F1_neg = 2(1-p)/(2-p)  =>  p = (2-2s)/(2-s), s = 4*s0
    """
    s = 4 * score_all_empty
    return (2 - 2 * s) / (2 - s)


def mean_shape_from_all_full(score_all_full: float, pos_ratio: float) -> float:
    """전부 1 제출 점수 s1 -> 양성 패치에서 '전부 1' 마스크의 평균 형상 F (= mean 2a/(1+a)).
    s1 = 0.5 * p/(1+p) + 0.5 * shape
    반환값 f 로부터 대략적인 평균 쓰레기 면적 비율 a ≈ f/(2-f) 를 얻습니다.
    """
    return 2 * score_all_full - pos_ratio / (1 + pos_ratio)
