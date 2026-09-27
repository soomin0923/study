#!/usr/bin/env bash
# 합성 데이터로 전체 파이프라인을 CPU 에서 한 바퀴 돌립니다 (코드 점검용).
#   bash tests/smoke_e2e.sh [작업폴더]
set -euo pipefail
cd "$(dirname "$0")/.."
W="${1:-$(mktemp -d)}"
echo "work dir: $W"

python tests/make_synthetic.py --out "$W/syn" --n 24
# 주최측 형식(arch 키 없음) 가짜 베이스라인 가중치
python - "$W" <<'PY'
import sys, torch
sys.path.insert(0, "submission/assets")
from cdkit.model import CDNet
torch.save({"state_dict": CDNet({}).net.state_dict()}, f"{sys.argv[1]}/fake_baseline.pt")
PY

python scripts/train.py --train "$W/syn/train:1.0" --val "$W/syn/val" --baseline "$W/fake_baseline.pt" \
  --cls-head --epochs 2 --samples-per-epoch 32 --bs 8 --workers 0 --lr 1e-3 --device cpu --out "$W/run"
python scripts/predict_folder.py --ckpt "$W/run/best.pt" --pairs "$W/syn/val" --out "$W/val.npz" --tta 2 --device cpu
python scripts/tune_rules.py --probs "$W/val.npz" --val "$W/syn/val" --out "$W/rules.json" --top 2
python scripts/visualize.py --probs "$W/val.npz" --val "$W/syn/val" --rules "$W/rules.json" --out "$W/vis"
python scripts/make_pseudo_labels.py --probs "$W/val.npz" --pairs "$W/syn/val" --out "$W/pseudo"

python scripts/build_submission.py --models "$W/run/best.pt" "$W/fake_baseline.pt" --rules "$W/rules.json" \
  --tta 2 --dist "$W" --name sub
# 제출물 리허설: zip 을 새 폴더에 풀어 그 폴더에서 실행 (채점 서버와 같은 방식)
python -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$W/sub.zip" "$W/unz"
(cd "$W/unz" && AIF_INPUT_DIR="$W/syn/eval" AIF_PREDICTION_PATH="$W/out/prediction.csv" \
  jupyter nbconvert --to notebook --execute predict.ipynb --output "$W/executed.ipynb" >/dev/null)
python scripts/validate_submission.py "$W/out/prediction.csv" --pairs "$W/syn/eval/pairs.csv"

# 검증셋을 대회 입력 형식으로 바꿔 노트북 결과를 정확 채점
mkdir -p "$W/val_eval/images"
python - "$W" <<'PY'
import sys, shutil, csv
from pathlib import Path
W = Path(sys.argv[1]); src = W / "syn" / "val"; dst = W / "val_eval"
ids = sorted(d.name for d in src.iterdir() if d.is_dir())
for i in ids:
    (dst / "images" / i).mkdir(parents=True, exist_ok=True)
    for f in ("pre.png", "post.png"):
        shutil.copy(src / i / f, dst / "images" / i / f)
with open(dst / "pairs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["id"]); w.writerows([[i] for i in ids])
PY
(cd "$W/unz" && AIF_INPUT_DIR="$W/val_eval" AIF_PREDICTION_PATH="$W/out/val_pred.csv" \
  jupyter nbconvert --to notebook --execute predict.ipynb --output "$W/executed_val.ipynb" >/dev/null)
python scripts/validate_submission.py "$W/out/val_pred.csv" --pairs "$W/val_eval/pairs.csv" --val "$W/syn/val"

python scripts/build_submission.py --probe empty --dist "$W" --name probe_empty >/dev/null
(cd "$W/probe_empty" && AIF_INPUT_DIR="$W/syn/eval" AIF_PREDICTION_PATH="$W/out/probe.csv" \
  jupyter nbconvert --to notebook --execute predict.ipynb --output "$W/probe.ipynb" >/dev/null)
python scripts/validate_submission.py "$W/out/probe.csv" --pairs "$W/syn/eval/pairs.csv"
if python scripts/build_submission.py --dist "$W" --name nomodel >/dev/null 2>&1; then
  test -f submission/assets/model/unet_r18_cd.pt || { echo "가중치 없는데 빌드 성공함"; exit 1; }
fi
echo "SMOKE OK"
