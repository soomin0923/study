"""제출 CSV 검증 (채점 실패 규칙) 과 정확 채점."""
import csv
from pathlib import Path

from . import CLASSES
from .geom import CellError, parse_cell, polygons_to_geometry

MAX_VERTS_FILE = 5_000_000


def read_ids(pairs_csv) -> list[str]:
    with Path(pairs_csv).open(encoding="utf-8-sig", newline="") as f:
        return [r["id"].strip() for r in csv.DictReader(f) if r.get("id", "").strip()]


def read_submission(pred_csv, ids: list[str] | None = None) -> dict:
    """채점 규칙대로 읽어 {class: {id: geometry}} 반환. 실패 사유가 있으면 ValueError."""
    with Path(pred_csv).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in ("id", *CLASSES) if c not in cols]
        if missing:
            raise ValueError(f"헤더에 열이 없습니다: {missing}")
        rows = list(reader)
    seen, total_v = set(), 0
    known = set(ids) if ids is not None else None
    geoms = {c: {} for c in CLASSES}
    for n, r in enumerate(rows, start=2):
        i = (r.get("id") or "").strip()
        if i in seen:
            raise ValueError(f"{n}행: 중복 id {i}")
        seen.add(i)
        if known is not None and i not in known:
            raise ValueError(f"{n}행: 목록에 없는 id {i}")
        for c in CLASSES:
            try:
                polys = parse_cell(r.get(c) or "")
            except CellError as e:
                raise ValueError(f"{n}행 {c}: {e}") from None
            total_v += sum(len(p) for p in polys)
            geoms[c][i] = polygons_to_geometry(polys)
    if total_v > MAX_VERTS_FILE:
        raise ValueError(f"파일 전체 정점 수 {total_v} > {MAX_VERTS_FILE}")
    return geoms
