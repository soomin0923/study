# 해안쓰레기 영역검출 (KNPS 2026 주제4)

0.3m 위성 RGB 패치(256×256)에서 해안쓰레기 집적대를 이진 마스크로 찾는 대회용 코드입니다.
점수 = 0.5 × macroF1(패치 존재 판정) + 0.5 × 형상 점수(1화소 허용 F-score, 정답 양성 패치 평균).

**핵심 전략**: 주최 측 베이스라인 가중치가 유일한 대회 도메인 자산입니다. 그래서 **① 산식에 맞춘 판정 규칙 튜닝 → ② 베이스라인에서 출발하는 파인튜닝** 순서로 진행합니다. 모든 결정은 직접 라벨링한 0.3m 검증셋으로 내립니다.

## 구성

`submission/` 폴더가 **그대로 제출 zip 의 내용**입니다 (zip 최상위 = 이 폴더 안).

```
submission/                     ← 제출물 템플릿 (대회 규격)
  predict.ipynb                 진입점. AIF_INPUT_DIR 읽기 → AIF_PREDICTION_PATH 쓰기
  requirements.txt              torch, segmentation-models-pytorch, numpy, opencv-python-headless
  assets/
    config.json                 모델 목록(assets 기준 상대경로), TTA, 시간 예산
    rules.json                  판정·후처리 규칙 (tune_rules.py 결과로 교체)
    model/                      가중치 (.pt, git 제외) ← unet_r18_debris_lite.pt 를 여기에
    debris_kit/                 공용 코드 (노트북과 학습 스크립트가 같이 씀)
      metric.py  rle.py  postprocess.py  model.py  infer.py
      losses.py  data.py  degrade.py
scripts/                        학습·분석·제출 빌드 (제출물에는 들어가지 않음)
  predict_folder.py  tune_rules.py  visualize.py  train.py  prepare_aihub.py
  make_pseudo_labels.py  build_submission.py  validate_submission.py  probe_calc.py
tests/                          단위 테스트, 합성 데이터 전체 점검(smoke_e2e.sh)
dist/                           build_submission.py 출력 (git 제외)
```

### 제출 zip 만들기
```bash
# 1) 가장 간단: 가중치를 템플릿 자리에 넣고 빌드
cp <베이스라인패키지>/assets/model/unet_r18_debris_lite.pt submission/assets/model/
python scripts/build_submission.py                      # -> dist/submission.zip

# 2) 학습한 모델·튜닝한 규칙으로 (템플릿은 건드리지 않고 dist/ 에 조립)
python scripts/build_submission.py --models runs/a/best.pt runs/b/best.pt --rules runs/rules.json --name sub_v2
```
- 가중치가 없으면 빌드가 멈춥니다 (`--allow-missing-weights` 로 구조만 묶을 수는 있지만 그 zip 은 제출하면 실패합니다).
- 노트북은 `!`/`%` 줄 없이 순수 파이썬이며, 경로는 환경변수와 `assets/` 상대경로만 씁니다.
- 셀은 setup / inputs / model / infer / save 로 나뉘어 있어 채점 로그에서 어느 단계에서 멈췄는지 보입니다.

## 데이터: 무엇을 어디에 쓰나

| 데이터 | 용도 | 필수 여부 |
|---|---|---|
| 베이스라인 가중치 `unet_r18_debris_lite.pt` | 규칙 튜닝 대상, 파인튜닝 초기값, 의사 라벨 교사 | **필수** |
| **직접 라벨링한 0.3m 검증셋** (150~300장) | 규칙 튜닝, 모델·데이터 선택의 유일한 기준 | **필수** |
| 0.3m급 비라벨 해안 영상 | 의사 라벨 → 학습 | 권장 |
| AIHub 국립공원 변화탐지 (드론 0.1m) | 열화 변환 후 학습 보조 + 해안 hard negative | 선택 (효과는 검증셋으로 확인) |

- AIHub 드론 데이터는 해상도·플랫폼이 달라서 **주 학습 자료가 아니라 보조 자료**입니다. 넣은 모델과 뺀 모델의 검증셋 점수를 비교해 쓸지 정하십시오.
- 0.3m 영상 출처는 라이선스가 대회 참가·파생물 제출을 허용하는 것만 쓰십시오 (예: 국토지리정보원 항공 정사영상, 공공누리 유형 확인). Google Earth 캡처는 약관상 불가합니다.
- 대회 자료(베이스라인 가중치·노트북, 평가 관련 자료)는 재배포 금지입니다. `data/`, `dist/`, `*.pt`, `*.zip` 은 `.gitignore` 로 제외되어 있으니 **레포에 올리지 마십시오.**

### 검증셋 라벨링 규칙 (대회 정답 규칙을 그대로 따름)
- 256×256 패치로 자르고, 쓰레기 조각이 덮는 화소를 1로 칠합니다. 띠·덩어리 단위이며 좁은 틈은 0이어도 됩니다.
- **칠하지 않는 것**: 밝은 자갈·조개껍질, 파도 거품·포말, 해조류·유목, 갯벌 사주·물고임, 선박·차량·어구.
- 빈 패치(쓰레기 없음)를 30~50% 포함합니다. 특히 위의 헷갈리는 대상이 있는 빈 패치를 많이 넣으십시오.
- 저장 형식: `data/val/images/<id>.png`, `data/val/masks/<id>.png` (0=배경, >0=쓰레기). CVAT·Label Studio 에서 마스크 PNG 로 내보내면 됩니다.

## 작업 순서

```bash
pip install -r requirements-dev.txt
# 베이스라인 패키지의 가중치를 submission/assets/model/ 에 둡니다 (커밋 금지)
BASE=submission/assets/model/unet_r18_debris_lite.pt
```

### 1. 리더보드 탐색 (제출 2회, 평가셋 분포 파악)
```bash
python scripts/build_submission.py --probe empty   # -> dist/probe_empty.zip   # 점수 s0
python scripts/build_submission.py --probe full    # -> dist/probe_full.zip    # 점수 s1
python scripts/probe_calc.py --empty <s0> --full <s1>                     # 양성 비율 p, 평균 면적
```

### 2. 베이스라인 + 규칙 튜닝 (재학습 없이 얻는 개선)
```bash
python scripts/predict_folder.py --ckpt $BASE --images data/val/images --out runs/val_base.npz --tta 8
python scripts/tune_rules.py --probs runs/val_base.npz --masks data/val/masks --out runs/rules_base.json --pos-ratio <p>
python scripts/visualize.py --probs runs/val_base.npz --images data/val/images --masks data/val/masks \
    --rules runs/rules_base.json --out runs/vis_base       # FP/FN 유형 확인
python scripts/build_submission.py --models $BASE --rules runs/rules_base.json --name sub_v1
```
`tune_rules.py` 는 베이스라인 동작(임계값 0.5)의 점수도 함께 출력하므로 개선폭을 바로 볼 수 있습니다.

### 3. 학습 데이터 준비
```bash
# (선택) AIHub 드론 → 0.3m. 라벨 형식과 해안쓰레기 클래스 값을 먼저 확인하십시오
python scripts/prepare_aihub.py --images <드론영상> --masks <클래스마스크> --debris-values <값> \
    --out data/aihub_sat --neg-out data/aihub_neg
# 비라벨 0.3m 영상 의사 라벨
python scripts/predict_folder.py --ckpt $BASE --images data/unlabeled/images --out runs/unl.npz
python scripts/make_pseudo_labels.py --probs runs/unl.npz --images data/unlabeled/images --out data/pseudo
```

### 4. 파인튜닝 (베이스라인 초기값 + 분류 헤드 + 허용오차 F 손실)
```bash
python scripts/train.py --train data/pseudo:2 data/aihub_sat:1 data/aihub_neg:1 \
    --val data/val --baseline $BASE --cls-head --epochs 30 --out runs/r18_cls
```
데이터 조합(가중치)을 바꿔 가며 검증 점수를 비교하십시오. 다른 인코더는 `--encoder efficientnet-b3` 처럼 지정합니다 (ImageNet 가중치에서 시작, 학습 PC 에 인터넷 필요).

### 5. 앙상블 + 규칙 재튜닝 + 제출
```bash
python scripts/predict_folder.py --ckpt runs/r18_cls/best.pt runs/b3/best.pt --images data/val/images --out runs/val_ens.npz
python scripts/tune_rules.py --probs runs/val_ens.npz --masks data/val/masks --out runs/rules_ens.json --pos-ratio <p>
python scripts/build_submission.py --models runs/r18_cls/best.pt runs/b3/best.pt --rules runs/rules_ens.json --name sub_v2
```
**규칙은 반드시 제출할 모델 조합과 같은 조합의 확률로 튜닝**하십시오 (앙상블하면 확률 분포가 바뀝니다).

### 로컬 리허설 (제출 전 필수)
```bash
cd dist/sub_v2
AIF_INPUT_DIR=../../data/val_as_eval AIF_PREDICTION_PATH=/tmp/pred.csv \
  jupyter nbconvert --to notebook --execute predict.ipynb --output /tmp/executed.ipynb
cd -
python scripts/validate_submission.py /tmp/pred.csv --patches data/val_as_eval/patches.csv --masks data/val/masks
```
`data/val_as_eval` 은 `patches.csv`(id 열) + `images/` 구조입니다. 네트워크를 끈 상태로 한 번 돌려 보는 것을 권합니다.

## 제출 노트북 동작
- `assets/config.json`(모델 목록, TTA, 시간 예산)과 `assets/rules.json`(판정 규칙)을 읽습니다. 가중치가 없으면 바로 명확한 오류로 멈춥니다.
- GPU 시험 추론에 실패하면 CPU 로 전환하고 TTA 를 줄입니다.
- 예상 소요 시간이 예산(기본 70분)을 넘으면 도중에 TTA 를 절반씩 줄입니다.
- 읽지 못한 패치는 빈 예측으로 내고 계속 진행합니다. 저장 후 채점 규칙대로 자체 검증합니다.

## 테스트
```bash
python -m pytest tests -q          # 산식·RLE·후처리·손실·체크포인트
bash tests/smoke_e2e.sh            # 합성 데이터로 학습→튜닝→제출 빌드→노트북 실행 전체 점검 (CPU 2분 내외)
```
