"""불법시설물 변화탐지(주제3) 공용 모듈.

추론 노트북(predict.ipynb)은 이 패키지를 assets/cdkit 로 동봉해 import 합니다.
추론 경로(geom, metric, postprocess, model, infer)는 torch/numpy/opencv/shapely 외의 의존성을 쓰지 않습니다.
"""

H = W = 256
CLASSES = ("new_building", "tree_removal")   # 모델 출력 채널 1, 2 (0 은 배경) - 베이스라인과 같은 순서
MIN_POS_AREA = 20.0                          # 채점 규칙: 폴리곤 합집합 면적이 20화소 미만이면 빈 예측
