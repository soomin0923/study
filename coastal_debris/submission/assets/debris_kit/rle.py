"""대회 규격 RLE 인코딩/디코딩과 제출 파일 검증."""
import csv
from pathlib import Path

import numpy as np

from . import H, W, N_PX


def rle_encode(mask: np.ndarray) -> str:
    """(H,W) 이진 마스크 -> 행 우선, 0-based '시작 길이' 쌍 문자열. 빈 마스크는 ''."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if m.size != N_PX:
        raise ValueError(f"mask size {m.size} != {N_PX}")
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return " ".join(f"{a} {b - a}" for a, b in zip(st.tolist(), en.tolist()))


def rle_decode(rle: str) -> np.ndarray:
    """채점 서버 규칙대로 해석합니다. 규칙 위반이면 ValueError (= 제출 전체 채점 실패)."""
    m = np.zeros(N_PX, dtype=np.uint8)
    rle = (rle or "").strip()
    if not rle:
        return m.reshape(H, W)
    toks = rle.split()
    if len(toks) % 2:
        raise ValueError("RLE 값 개수가 홀수입니다")
    try:
        vals = [int(t) for t in toks]
    except ValueError as e:
        raise ValueError(f"정수가 아닌 RLE 값: {e}") from None
    for s, n in zip(vals[0::2], vals[1::2]):
        if n < 1:
            raise ValueError(f"길이가 1 미만인 구간: {s} {n}")
        if s < 0 or s + n > N_PX:
            raise ValueError(f"0~{N_PX - 1} 범위를 벗어난 구간: {s} {n}")
        m[s:s + n] = 1
    return m.reshape(H, W)


def read_ids(patches_csv: Path) -> list[str]:
    with Path(patches_csv).open(encoding="utf-8-sig", newline="") as f:
        return [r["id"].strip() for r in csv.DictReader(f) if r.get("id", "").strip()]


def validate_submission(pred_csv: Path, patches_csv: Path | None = None) -> dict:
    """제출 CSV 를 채점 규칙대로 검사합니다. 실패 사유가 있으면 ValueError."""
    with Path(pred_csv).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or [c.strip() for c in reader.fieldnames] != ["id", "rle"]:
            raise ValueError(f"헤더가 id,rle 가 아닙니다: {reader.fieldnames}")
        rows = list(reader)
    seen, n_pos, n_empty_small = set(), 0, 0
    for r in rows:
        i = r["id"].strip()
        if i in seen:
            raise ValueError(f"중복 id: {i}")
        seen.add(i)
        m = rle_decode(r["rle"] or "")
        s = int(m.sum())
        if s >= 50:
            n_pos += 1
        elif s > 0:
            n_empty_small += 1
    info = {"rows": len(rows), "positive(>=50px)": n_pos, "nonempty_but_<50px": n_empty_small}
    if patches_csv is not None:
        ids = set(read_ids(patches_csv))
        extra = seen - ids
        if extra:
            raise ValueError(f"목록에 없는 id {len(extra)}개 (예: {sorted(extra)[:3]})")
        info["missing_ids(treated_empty)"] = len(ids - seen)
    return info
