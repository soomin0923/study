"""제출 CSV 를 채점 규칙대로 검사합니다. 실패하면 종료코드 1.

  python scripts/validate_submission.py prediction.csv --patches <입력>/patches.csv
  # 정답 마스크가 있으면 점수까지 계산 (로컬 리허설)
  python scripts/validate_submission.py prediction.csv --patches data/val/patches.csv --masks data/val/masks
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from _common import ROOT  # noqa: F401
from debris_kit.data import read_mask
from debris_kit.metric import competition_score
from debris_kit.rle import read_ids, rle_decode, validate_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--patches", default=None)
    ap.add_argument("--masks", default=None)
    a = ap.parse_args()
    try:
        info = validate_submission(Path(a.csv), Path(a.patches) if a.patches else None)
    except ValueError as e:
        print("채점 실패 사유:", e)
        sys.exit(1)
    print("OK", info)
    if a.masks and a.patches:
        with open(a.csv, encoding="utf-8-sig", newline="") as f:
            pred = {r["id"].strip(): r["rle"] for r in csv.DictReader(f)}
        ids = read_ids(Path(a.patches))
        preds = [rle_decode(pred.get(i, "")) for i in ids]
        gts = [read_mask(Path(a.masks) / f"{i}.png") for i in ids]
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in competition_score(preds, gts).items()})


if __name__ == "__main__":
    main()
