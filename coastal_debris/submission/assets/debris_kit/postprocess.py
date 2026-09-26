"""확률 지도 -> 제출 마스크 변환 규칙.

두 단계로 나눕니다. 규칙 탐색(tune_rules.py)에서 두 단계를 따로 캐시해 빠르게 격자 탐색하기 위함입니다.
1) make_mask : 화소 규칙 (임계값, 작은 연결요소 제거, closing, 50화소 보장)
2) gate      : 패치 판정 규칙 (분류 헤드 확률, 분할 최대 확률, 면적)

rules 예시 (assets/rules.json 에 저장):
{
  "thr_abs": 0.5,       화소 임계값 상한
  "thr_rel": 1.0,       화소 임계값 = min(thr_abs, thr_rel * 패치 최대확률)  (확신도 낮은 패치에서 영역을 살림)
  "min_cc": 0,          이 면적 미만 연결요소 제거 (전부 지워지면 가장 큰 것 하나는 남김)
  "close": 0,           closing 커널 크기 (0 이면 끔)
  "guarantee_px": 60,   양성 판정 패치의 마스크가 이보다 작으면 확장 (50 미만이면 채점상 음성이라서)
  "t_max": 0.5,         분할 최대확률이 이 이상이고
  "min_area": 50,       후처리 면적이 이 이상이면 양성
  "t_cls": null         분류 헤드 확률이 이 이상이면 양성 (null 이면 분류 헤드 미사용)
}
"""
import cv2
import numpy as np

from . import MIN_POS_PX

DEFAULT_RULES = {
    "thr_abs": 0.5, "thr_rel": 1.0, "min_cc": 0, "close": 0, "guarantee_px": 60,
    "t_max": 0.5, "min_area": MIN_POS_PX, "t_cls": None,
}
BASELINE_RULES = {**DEFAULT_RULES, "guarantee_px": 0}  # 베이스라인 노트북과 같은 동작

_K3 = np.ones((3, 3), np.uint8)


def _remove_small_cc(m: np.ndarray, min_cc: int) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= min_cc) + 1
    if keep.size == 0:
        keep = np.array([int(np.argmax(areas)) + 1])
    return np.isin(lab, keep).astype(np.uint8)


def _grow_to(m: np.ndarray, prob: np.ndarray, target: int) -> np.ndarray:
    """마스크(없으면 최대확률 지점)에서 시작해 target 화소 이상이 될 때까지 확장.
    확장 후보 중 확률이 높은 화소부터 채워 target 을 약간만 넘기도록 합니다."""
    if not m.any():
        m = np.zeros_like(m)
        m.flat[int(np.argmax(prob))] = 1
    for _ in range(64):
        s = int(m.sum())
        if s >= target:
            break
        ring = (cv2.dilate(m, _K3) > 0) & (m == 0)
        need = target - s
        cand = np.flatnonzero(ring)
        if cand.size > need:
            cand = cand[np.argsort(-prob.flat[cand], kind="stable")[:need]]
        m = m.copy()
        m.flat[cand] = 1
    return m


def pixel_threshold(maxp: float, rules: dict) -> float:
    return min(float(rules["thr_abs"]), float(rules["thr_rel"]) * maxp)


def make_mask(prob: np.ndarray, rules: dict) -> tuple[np.ndarray, int]:
    """(마스크, 확장 전 면적). 판정과 무관하게 '양성이라면 낼 마스크'를 만듭니다."""
    maxp = float(prob.max())
    m = (prob >= pixel_threshold(maxp, rules)).astype(np.uint8)
    if rules.get("close", 0) and m.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(rules["close"]),) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if rules.get("min_cc", 0) and m.any():
        m = _remove_small_cc(m, int(rules["min_cc"]))
    area = int(m.sum())
    g = int(rules.get("guarantee_px", 0) or 0)
    if g and area < g:
        m = _grow_to(m, prob, g)
    return m, area


def gate(maxp: float, area: int, cls_prob: float | None, rules: dict) -> bool:
    seg_pos = maxp >= rules["t_max"] and area >= rules["min_area"]
    t_cls = rules.get("t_cls")
    cls_pos = t_cls is not None and cls_prob is not None and cls_prob >= t_cls
    return bool(seg_pos or cls_pos)


def decide(prob: np.ndarray, cls_prob: float | None, rules: dict) -> np.ndarray:
    """최종 제출 마스크. 음성이면 빈 마스크, 양성이어도 50화소 미만이면 빈 마스크로 정리."""
    m, area = make_mask(prob, rules)
    if not gate(float(prob.max()), area, cls_prob, rules) or int(m.sum()) < MIN_POS_PX:
        return np.zeros(prob.shape, np.uint8)
    return m
