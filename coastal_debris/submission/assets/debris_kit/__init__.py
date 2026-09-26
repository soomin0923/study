"""해안쓰레기 영역검출 대회용 공용 모듈.

추론 노트북(predict.ipynb)은 이 패키지를 assets/debris_kit 로 동봉해 import 합니다.
따라서 추론 경로(rle, metric, postprocess, model, tta)는 torch/numpy/opencv 외의 의존성을 쓰지 않습니다.
"""

H = W = 256
N_PX = H * W
MIN_POS_PX = 50  # 채점 규칙: 1인 화소가 50개 미만이면 빈 예측
