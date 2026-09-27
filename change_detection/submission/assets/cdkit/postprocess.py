"""확률 지도 -> 클래스별 제출 폴리곤.

단계
1) 무영상 영역 제거: 전/후 어느 한쪽이라도 검정(0,0,0)인 큰 영역은 판정 대상이 아니므로 확률을 0으로
2) make_mask : 화소 임계값, closing, 작은 연결요소 제거, 20화소 보장
3) gate      : 쌍 단위 존재 판정 (분류 헤드 확률 / 분할 최대 확률 / 면적)
4) 폴리곤화   : 화소 경계를 따라 외곽선 -> simplify

rules.json 은 클래스별 규칙을 담습니다: {"new_building": {...}, "tree_removal": {...}}
"""
import cv2
import numpy as np

from . import CLASSES, MIN_POS_AREA
from .geom import mask_to_polygons, polygons_to_geometry

DEFAULT_CLASS_RULES = {
    "thr_abs": 0.5,        # 화소 임계값 상한
    "thr_rel": 1.0,        # 화소 임계값 = min(thr_abs, thr_rel * 패치 최대확률)
    "close": 0,            # closing 커널 (0=끔). 증축은 곧은 변이라 작게, 벌목은 크게 쓰는 경향
    "min_cc": 30,          # 이 면적 미만 조각 제거 (베이스라인과 같은 30). 전부 지워지면 가장 큰 것 하나는 남김
    "guarantee_px": 24,    # 양성 판정인데 이보다 작으면 확장 (20 미만이면 채점상 음성이라서)
    "t_max": 0.5,          # 분할 최대확률이 이 이상이고
    "min_area": 20,        # 후처리 면적이 이 이상이면 양성
    "t_cls": None,         # 분류 헤드 확률이 이 이상이면 양성 (None 이면 미사용)
    "simplify": 0.5,       # 외곽선 단순화 허용오차(px). 채점 허용오차 1px 안
}
DEFAULT_RULES = {c: dict(DEFAULT_CLASS_RULES) for c in CLASSES}
BASELINE_RULES = {c: {**DEFAULT_CLASS_RULES, "guarantee_px": 0} for c in CLASSES}

_K3 = np.ones((3, 3), np.uint8)


def nodata_mask(img: np.ndarray, min_area: int = 64) -> np.ndarray:
    """검정(0,0,0) 화소 중 일정 면적 이상 뭉친 영역 = 무영상. 어두운 그림자 속 몇 화소짜리 0 은 제외."""
    z = (img.reshape(img.shape[0], img.shape[1], -1).max(axis=2) == 0).astype(np.uint8)
    if not z.any():
        return z.astype(bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(z, connectivity=8)
    keep = np.flatnonzero(st[1:, cv2.CC_STAT_AREA] >= min_area) + 1
    return np.isin(lab, keep)


def valid_mask(pre: np.ndarray, post: np.ndarray, margin: int = 1) -> np.ndarray:
    """두 영상 모두 검지 않은 구역 (경계 margin 화소는 안전하게 제외)."""
    nd = nodata_mask(pre) | nodata_mask(post)
    if margin and nd.any():
        nd = cv2.dilate(nd.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)) > 0
    return ~nd


def _remove_small_cc(m, min_cc):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_cc) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(areas)) + 1])
    return np.isin(lab, keep).astype(np.uint8)


def _grow_to(m, prob, target, valid):
    if not m.any():
        m = np.zeros_like(m)
        m.flat[int(np.argmax(prob * valid))] = 1
    for _ in range(64):
        s = int(m.sum())
        if s >= target:
            break
        ring = (cv2.dilate(m, _K3) > 0) & (m == 0) & valid
        cand = np.flatnonzero(ring)
        if cand.size == 0:
            break
        need = target - s
        if cand.size > need:
            cand = cand[np.argsort(-prob.flat[cand], kind="stable")[:need]]
        m = m.copy()
        m.flat[cand] = 1
    return m


def make_mask(prob: np.ndarray, r: dict, valid: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """(마스크, 확장 전 면적). prob 는 이미 무영상 영역이 0 이라고 가정해도 되고, valid 로 다시 막습니다."""
    if valid is None:
        valid = np.ones(prob.shape, bool)
    maxp = float(prob.max())
    thr = min(float(r["thr_abs"]), float(r["thr_rel"]) * maxp)
    m = ((prob >= thr) & valid & (prob > 0)).astype(np.uint8)
    if r.get("close") and m.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(r["close"]),) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k) & valid.astype(np.uint8)
    if r.get("min_cc") and m.any():
        m = _remove_small_cc(m, int(r["min_cc"]))
    area = int(m.sum())
    g = int(r.get("guarantee_px") or 0)
    if g and area < g:
        m = _grow_to(m, prob, g, valid)
    return m, area


def gate(maxp: float, area: int, cls_prob, r: dict) -> bool:
    seg_pos = maxp >= r["t_max"] and area >= r["min_area"]
    cls_pos = r.get("t_cls") is not None and cls_prob is not None and cls_prob >= r["t_cls"]
    return bool(seg_pos or cls_pos)


def decide_class(prob, cls_prob, r, valid=None) -> np.ndarray:
    if valid is not None:
        prob = prob * valid
    m, area = make_mask(prob, r, valid)
    if not gate(float(prob.max()), area, cls_prob, r) or m.sum() < MIN_POS_AREA:
        return np.zeros(prob.shape, np.uint8)
    return m


def decide(probs: np.ndarray, cls_probs, rules: dict, valid=None) -> dict:
    """probs: (2,H,W) [증축, 벌목] -> {class: 폴리곤 리스트}."""
    out = {}
    for k, c in enumerate(CLASSES):
        r = {**DEFAULT_CLASS_RULES, **rules.get(c, {})}
        cp = None if cls_probs is None else float(cls_probs[k])
        m = decide_class(probs[k], cp, r, valid)
        polys = mask_to_polygons(m, simplify_px=float(r.get("simplify", 0.5))) if m.any() else []
        if polys and polygons_to_geometry(polys).area < MIN_POS_AREA:   # 단순화로 20 밑으로 줄어든 경우
            polys = []
        out[c] = polys
    return out
