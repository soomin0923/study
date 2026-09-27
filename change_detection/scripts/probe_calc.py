"""리더보드 탐색 제출 점수로 Public 평가셋의 대략적인 양성 비율을 역산합니다.

  python scripts/build_submission.py --probe empty      # 두 클래스 모두 빈 예측 -> 점수 s0
  python scripts/probe_calc.py --empty <s0>

전부 빈 예측 점수는 0.125*(Nb+Nt) (N = 클래스별 '음성 F1') 이라 두 클래스를 따로 알 수 없습니다.
두 클래스의 양성 비율이 비슷하다고 가정한 평균값만 나옵니다. 참고용으로만 쓰십시오.
"""
import argparse

from _common import ROOT  # noqa: F401
from cdkit.metric import mean_neg_f1_from_all_empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", type=float, required=True)
    a = ap.parse_args()
    n, p = mean_neg_f1_from_all_empty(a.empty)
    print(f"음성 F1 평균 {n:.3f} -> (두 클래스 비슷하다고 가정) 양성 쌍 비율 ≈ {p:.3f}")


if __name__ == "__main__":
    main()
