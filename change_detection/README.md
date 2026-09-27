# 불법시설물 변화탐지 (KNPS 2026 주제3)

같은 땅을 두 시점에 찍은 위성영상 전/후 쌍(256×256, 1화소 0.55~0.7m)에서 **증축(new_building)** 과 **벌목(tree_removal)** 을 후 영상 위 폴리곤으로 찾는 대회용 코드입니다.

- 클래스 점수 = 0.5 × macroF1(쌍 단위 존재 판정) + 0.5 × 형상 점수(1화소 허용 F-score)
- 최종 점수 = 0.5 × 증축 점수 + 0.5 × 벌목 점수
- **학습·샘플·평가 영상을 하나도 주지 않습니다.** 베이스라인 가중치(`unet_r18_cd.pt`)만 제공됩니다.

**핵심 전략**: 베이스라인 가중치가 유일한 대회 도메인 자산입니다. **① 산식에 맞춘 판정 규칙 튜닝 → ② 외부 데이터·합성 변화로 파인튜닝** 순서로 진행하고, 모든 결정은 직접 만든 검증셋으로 내립니다.

## 구성

`submission/` 폴더가 **그대로 제출 zip 의 내용**입니다.

```
submission/
  predict.ipynb            진입점 (scripts/_notebook_src.py 로 생성)
  requirements.txt         torch, segmentation-models-pytorch, numpy, opencv-python-headless, shapely
  assets/
    config.json            모델 목록, TTA, 시간 예산
    rules.json             클래스별 판정·후처리 규칙
    model/                 가중치 자리 (git 제외) ← unet_r18_cd.pt
    cdkit/                 공용 코드
      geom.py              마스크↔폴리곤, 제출 셀 JSON 해석(채점 실패 규칙)
      metric.py            산식 (폴리곤 정확 계산 + 래스터 근사)
      postprocess.py       무영상 제거, 임계값, 조각 제거, 20화소 보장, 판정, 폴리곤화
      model.py infer.py    6채널 UNet(+분류 헤드), 앙상블·D4 TTA
      losses.py data.py    학습 손실, 데이터셋·'변화 아닌 차이' augmentation
      submission.py        제출 CSV 검증
scripts/
  predict_folder.py  tune_rules.py  visualize.py  train.py  make_pseudo_labels.py
  prepare_aihub.py (pair: 실제 전/후 쌍 / synth: 한 시점 영상에서 변화 합성)
  build_submission.py  validate_submission.py  probe_calc.py  _notebook_src.py
notebooks/workflow.ipynb   학습 → zip → 리허설 → (디버그)제출을 한 노트북에서
tests/  test_kit.py  smoke_e2e.sh  make_synthetic.py
```

## 기본 동작 = 주최 측 베이스라인과 같음

`submission/assets/rules.json` 기본값과 TTA 1 은 주최 측 베이스라인 노트북과 같은 결과를 냅니다
(3클래스 argmax → 30화소 미만 조각 제거 → 합친 면적 20 미만이면 빈 예측). 합성 60쌍에서 판정 60/60 일치,
폴리곤 IoU 평균 0.999 로 확인했습니다. 차이는 무영상 경계 1화소 여유뿐입니다.

- 베이스라인 가중치 메타데이터에 `steps: 100` 이 적혀 있습니다. 100 스텝만 학습한 '파이프라인 확인용' 모델이라
  성능 기준선으로 보기 어렵습니다.
- TTA 를 켜면(8) 이 모델은 방향마다 답이 달라 평균 후 양성 판정이 크게 줄었습니다(합성 60쌍에서 52 → 6).
  TTA·임계값은 검증셋이 생긴 뒤 `tune_rules.py` 와 비교로 정하십시오.

## 데이터 폴더 규약

```
data/<소스>/<id>/pre.png  post.png  building.png  tree.png  [ignore.png]
```
마스크는 0=배경, >0=해당 변화. 마스크 파일이 없으면 그 클래스는 '변화 없음'.

## 환경 설정

Python 3.11 을 권장합니다 (torch>=2.5, numpy 2 와 모두 호환). 채점 서버는 제출할 때 `requirements.txt` 로
최신 torch·smp 를 설치하므로, 로컬도 같은 메이저 버전(torch 2.x, smp 0.5.x)을 쓰면 됩니다.

**NVIDIA GPU (Windows/Linux)** - 학습까지 할 때
```bash
conda create -n knps python=3.11 -y && conda activate knps          # 또는 uv venv --python 3.11
nvidia-smi                                                           # 드라이버의 CUDA 버전 확인
pip install torch --index-url https://download.pytorch.org/whl/cu126 # 드라이버에 맞는 휠 (pytorch.org 참고)
pip install -r requirements-dev.txt
python -m ipykernel install --user --name knps --display-name "KNPS (py3.11)"
python -c "import torch; print(torch.cuda.is_available())"           # True 여야 GPU 학습
```

**GPU 없음 / Mac** - 규칙 튜닝·제출까지는 CPU 로 충분 (학습은 느림). 같은 방법에서 torch 휠만 기본으로 설치.

**Google Colab** - 무료 GPU. `aifactory` 가 Colab 을 지원하고, 참여키는 Colab Secrets 에 `AIF_API_KEY` 로 넣으면 됩니다.

### 제출 도구(aifactory) 동작 - 꼭 알아둘 것
- `%aifactory submit` 은 **그 셀을 실행한 노트북 자체**를 `predict.ipynb` 로 올리고, **현재 폴더의** `requirements.txt` 와
  `assets/` 만 함께 올립니다. 학습 노트북에서 그냥 실행하면 학습 노트북이 채점 서버에서 돌아갑니다.
  → `notebooks/workflow.ipynb` 처럼 `dist/<이름>/` 로 이동해 `--notebook predict.ipynb` 로 지정하십시오.
- 참여키는 노트북에 적지 않습니다. 환경변수 `AIF_API_KEY` 가 없으면 입력창이 뜹니다 (노트북은 서버로 올라갑니다).
- `--debug` : 공개 디버그 데이터로 실행되고 실행 로그를 제출 이력에서 볼 수 있습니다. 리더보드 미반영, 하루 횟수 제한.
  새 코드는 먼저 디버그 제출로 서버에서 도는지 확인하십시오.
- 로컬 Jupyter 에서는 **디스크에 저장된** 노트북이 올라갑니다. 저장 후 제출하십시오.
- 터미널에서도 됩니다: `cd dist/<이름> && aifactory submit --notebook predict.ipynb --model-name <이름> [--debug]`

## 작업 순서

```bash
# (환경 설정 후)
cp <베이스라인패키지>/assets/model/unet_r18_cd.pt submission/assets/model/
BASE=submission/assets/model/unet_r18_cd.pt
```

1. **검증셋 만들기** (필수): 0.5~0.7m급 전/후 쌍을 구해 `data/val/<id>/` 에 두고 증축·벌목을 직접 칠합니다.
2. **베이스라인 + 규칙 튜닝 → 첫 제출**
   ```bash
   python scripts/predict_folder.py --ckpt $BASE --pairs data/val --out runs/val_base.npz
   python scripts/tune_rules.py --probs runs/val_base.npz --val data/val --out runs/rules_base.json
   python scripts/visualize.py --probs runs/val_base.npz --val data/val --rules runs/rules_base.json --out runs/vis_base
   python scripts/build_submission.py --models $BASE --rules runs/rules_base.json --name sub_v1
   ```
3. **학습 데이터**
   ```bash
   python scripts/prepare_aihub.py pair  --pre ... --post ... --labels ... --building-values .. --tree-values .. --out data/aihub_cd
   python scripts/prepare_aihub.py synth --images ... --labels ... --building-values .. --forest-values .. --out data/synth_cd
   python scripts/predict_folder.py --ckpt $BASE --pairs data/unlabeled --out runs/unl.npz
   python scripts/make_pseudo_labels.py --probs runs/unl.npz --pairs data/unlabeled --out data/pseudo
   ```
4. **파인튜닝**
   ```bash
   python scripts/train.py --train data/aihub_cd:2 data/synth_cd:1 data/pseudo:1 --val data/val \
       --baseline $BASE --cls-head --epochs 30 --out runs/r18_cls
   ```
5. **앙상블 + 규칙 재튜닝 + 제출** (규칙은 제출할 모델 조합 그대로 다시 튜닝)
6. **리허설**: zip 을 풀어 그 폴더에서 `jupyter nbconvert --execute` 후
   `python scripts/validate_submission.py <csv> --pairs <pairs.csv> [--val data/val]` 로 채점 실패 규칙 검사와 정확 채점.

## 테스트
```bash
python -m pytest tests -q
bash tests/smoke_e2e.sh
```
