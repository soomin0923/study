"""마스크 <-> 폴리곤 변환과 제출 셀(JSON) 해석.

좌표 규약: 후 영상 패치의 픽셀좌표, 원점 왼쪽 위 모서리. 화소 (r, c) 는 [c, c+1] x [r, r+1] 을 차지합니다.
"""
import json
import math

import cv2
import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import unary_union

from . import H, W

PATCH = box(0, 0, W, H)
MAX_VERTS_CELL = 262_144


def _row_boxes(m: np.ndarray):
    """행 단위 런을 화소 경계 사각형으로. 합집합하면 마스크 외곽선을 그대로 따르는 도형이 됩니다."""
    out = []
    for r in np.flatnonzero(m.any(axis=1)):
        pad = np.concatenate(([0], m[r].astype(np.int8), [0]))
        e = np.flatnonzero(np.diff(pad))
        out += [box(float(s), float(r), float(t), float(r + 1)) for s, t in zip(e[::2], e[1::2])]
    return out


def _polys(g) -> list[Polygon]:
    if g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, (MultiPolygon, GeometryCollection)):
        return [p for x in g.geoms for p in _polys(x)]
    return []


def mask_to_polygons(mask: np.ndarray, simplify_px: float = 0.5, ndigits: int = 2) -> list:
    """(H,W) 이진 마스크 -> [[[x,y],...], ...]. 구멍은 버리고 바깥 경계만 씁니다 (제출 형식에 구멍이 없음)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    out = []
    # 연결요소별로 처리하면 거대한 합집합 한 번보다 빠르고 안정적입니다
    n, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
    for k in range(1, n):
        comp = lab == k
        ys, xs = np.nonzero(comp)
        y0, x0 = ys.min(), xs.min()
        sub = comp[y0:ys.max() + 1, x0:xs.max() + 1]
        g = unary_union(_row_boxes(sub))
        for p in _polys(g):
            p = Polygon(p.exterior)                   # 구멍 제거
            if simplify_px > 0:
                p = p.simplify(simplify_px, preserve_topology=True)
            if p.is_empty or p.area <= 0:
                continue
            pts = list(p.exterior.coords)[:-1]
            out.append([[round(float(x) + x0, ndigits), round(float(y) + y0, ndigits)] for x, y in pts])
    return out


def polygons_to_cell(polys: list) -> str:
    return json.dumps(polys, separators=(",", ":")) if polys else ""


class CellError(ValueError):
    """채점 서버가 '제출 전체 실패'로 처리하는 셀."""


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_cell(cell: str) -> list:
    """셀 문자열 -> 폴리곤(점 목록) 리스트. 채점 실패 사유면 CellError."""
    s = (cell or "").strip()
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise CellError(f"JSON 으로 읽을 수 없음: {e}") from None
    if not isinstance(data, list):
        raise CellError("폴리곤 목록(배열)이 아님")
    nv = 0
    for poly in data:
        if not isinstance(poly, list):
            raise CellError("폴리곤이 점 목록이 아님 (점 목록만 넣었거나 객체)")
        for pt in poly:
            if not (isinstance(pt, list) and len(pt) == 2):
                raise CellError("점이 [x, y] 가 아님 (배열이 한 겹 더 감싸였거나 점 목록만 넣음)")
            if not (_num(pt[0]) and _num(pt[1])):
                raise CellError(f"숫자가 아닌 좌표 또는 NaN: {pt}")
        nv += len(poly)
    if nv > MAX_VERTS_CELL:
        raise CellError(f"셀 정점 수 {nv} > {MAX_VERTS_CELL}")
    return data


def cell_vertex_count(cell: str) -> int:
    return sum(len(p) for p in parse_cell(cell))


def polygons_to_geometry(polys: list):
    """채점 규칙대로 합집합 도형을 만듭니다: 3점 미만/면적 0 무시, 자기교차 복구, 패치 밖 잘라냄."""
    geoms = []
    for pts in polys:
        if len(pts) < 3:
            continue
        p = Polygon(pts)
        if not p.is_valid:
            p = make_valid(p)
        if p.is_empty or p.area <= 0:
            continue
        geoms.append(p)
    if not geoms:
        return Polygon()
    g = unary_union(geoms).intersection(PATCH)
    return unary_union(_polys(g)) if not g.is_empty else Polygon()


def rasterize(polys: list, supersample: int = 4) -> np.ndarray:
    """폴리곤 -> (H,W) uint8 마스크 (화소 중심 기준, 로컬 학습 라벨 생성용)."""
    m = np.zeros((H * supersample, W * supersample), np.uint8)
    for pts in polys:
        if len(pts) >= 3:
            a = np.round(np.asarray(pts, np.float64) * supersample * 16).astype(np.int32)
            cv2.fillPoly(m, [a], 1, lineType=cv2.LINE_8, shift=4)
    if supersample == 1:
        return m
    return (cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
