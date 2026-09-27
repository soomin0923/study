"""제출 CSV 를 채점 규칙대로 검사하고, 정답이 있으면 폴리곤 정확 계산으로 점수를 냅니다.

  python scripts/validate_submission.py prediction.csv --pairs <입력>/pairs.csv
  python scripts/validate_submission.py prediction.csv --pairs data/val_eval/pairs.csv --val data/val
"""
import argparse
import sys
from pathlib import Path

from _common import ROOT  # noqa: F401
from cdkit import CLASSES, MIN_POS_AREA
from cdkit.data import read_mask
from cdkit.geom import mask_to_polygons, polygons_to_geometry
from cdkit.metric import score_geoms
from cdkit.submission import read_ids, read_submission

FILES = {"new_building": "building.png", "tree_removal": "tree.png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--val", default=None, help="<id>/building.png, tree.png 정답 폴더")
    a = ap.parse_args()
    ids = read_ids(a.pairs) if a.pairs else None
    try:
        geoms = read_submission(a.csv, ids)
    except ValueError as e:
        print("채점 실패 사유:", e)
        sys.exit(1)
    info = {c: sum(1 for g in geoms[c].values() if g.area >= MIN_POS_AREA) for c in CLASSES}
    print("OK 행", len(geoms[CLASSES[0]]), "양성(>=20px²)", info,
          "누락 id(빈 예측 처리)", (len(set(ids) - set(geoms[CLASSES[0]])) if ids else "-"))
    if a.val and ids:
        gt = {c: {} for c in CLASSES}
        for i in ids:
            for c in CLASSES:
                p = Path(a.val) / i / FILES[c]
                m = read_mask(p) if p.exists() else None
                gt[c][i] = polygons_to_geometry(mask_to_polygons(m, simplify_px=0)) if m is not None else None
        r = score_geoms(geoms, gt, ids)
        print(f"점수 {r['score']:.4f}  " + "  ".join(
            f"{c}: {r[c]['score']:.4f} (f1 {r[c]['f1']:.3f}, shape {r[c]['shape']:.3f}, "
            f"fp {r[c]['fp']:.0f}, fn {r[c]['fn']:.0f})" for c in CLASSES))


if __name__ == "__main__":
    main()
