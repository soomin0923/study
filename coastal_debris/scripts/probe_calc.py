"""리더보드 탐색 제출 점수로 Public 평가셋 통계를 역산합니다.

제출물 만들기 (build_submission.py --probe):
  --probe empty : 모든 패치를 빈 마스크로 제출  -> 점수 s0
  --probe full  : 모든 패치를 전부 1 로 제출    -> 점수 s1

  python scripts/probe_calc.py --empty 0.1234 --full 0.3456
"""
import argparse

from _common import ROOT  # noqa: F401
from debris_kit.metric import mean_shape_from_all_full, pos_ratio_from_all_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", type=float, required=True, help="전부 빈 마스크 제출 점수")
    ap.add_argument("--full", type=float, default=None, help="전부 1 제출 점수")
    a = ap.parse_args()
    p = pos_ratio_from_all_empty(a.empty)
    print(f"양성 패치 비율 p ≈ {p:.3f}  -> tune_rules.py --pos-ratio {p:.3f}")
    if a.full is not None:
        f = mean_shape_from_all_full(a.full, p)
        area = f / (2 - f) if 0 < f < 2 else float("nan")
        print(f"'전부 1' 마스크의 평균 형상 F ≈ {f:.3f} -> 양성 패치의 쓰레기 면적 비율 대략 {area:.3f} "
              f"(≈ {area * 65536:.0f} 화소, 1화소 확장 포함, 평균의 역산이라 대략값)")


if __name__ == "__main__":
    main()
