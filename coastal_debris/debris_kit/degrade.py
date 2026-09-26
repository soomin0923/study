"""고해상도(드론 0.1m) 영상 -> 0.3m 위성 영상 흉내 변환.

주최 측 안내대로 드론 영상은 해상도·플랫폼이 달라 그대로는 학습 자료로 맞지 않습니다.
여기서는 광학계 흐림 + 면적 평균 축소 + 노이즈/압축으로 '위성처럼 보이게' 만든 뒤 사용합니다.
이 데이터가 실제로 도움이 되는지는 반드시 직접 만든 0.3m 검증셋 점수로 판단하십시오.
"""
import cv2
import numpy as np


def degrade_image(img: np.ndarray, factor: float, rng) -> np.ndarray:
    """img: (h,w,3) uint8 RGB, factor = dst_gsd / src_gsd (0.1m -> 0.3m 이면 3)."""
    h, w = img.shape[:2]
    out = (int(round(w / factor)), int(round(h / factor)))
    x = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(0.35, 0.65) * factor)   # 위성 MTF 근사
    x = cv2.resize(x, out, interpolation=cv2.INTER_AREA)
    x = np.clip(x + rng.normal(0, rng.uniform(1, 4), x.shape), 0, 255).astype(np.uint8)
    _, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return np.ascontiguousarray(cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1])


def degrade_mask(mask: np.ndarray, factor: float, frac_thr: float = 0.4, close: int = 0) -> np.ndarray:
    """면적 비율로 축소 후 임계값. 한 화소(0.3m)를 충분히 덮지 못한 조각은 사라집니다(대회 최소 크기 규칙).
    close>0 이면 개체 단위 라벨을 집적대(띠·덩어리) 단위에 가깝게 잇습니다."""
    h, w = mask.shape
    out = (int(round(w / factor)), int(round(h / factor)))
    m = cv2.resize((mask > 0).astype(np.float32), out, interpolation=cv2.INTER_AREA)
    m = (m >= frac_thr).astype(np.uint8)
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m
