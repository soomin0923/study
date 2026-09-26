#!/usr/bin/env bash
# 합성 데이터로 전체 파이프라인을 CPU 에서 한 바퀴 돌립니다 (코드 점검용, 수 분 소요).
#   bash tests/smoke_e2e.sh [작업폴더]
set -euo pipefail
cd "$(dirname "$0")/.."
W="${1:-$(mktemp -d)}"
echo "work dir: $W"

python tests/make_synthetic.py --out "$W/syn" --n 24
# 주최측 형식(arch 키 없음) 가짜 베이스라인 가중치
python - "$W" <<'EOF'
import sys, torch
sys.path.insert(0, "submission/assets")
from debris_kit.model import DebrisNet
torch.save({"state_dict": DebrisNet({}).net.state_dict()}, f"{sys.argv[1]}/fake_baseline.pt")
EOF

python scripts/train.py --train "$W/syn/train:1.0" --val "$W/syn/val" --baseline "$W/fake_baseline.pt" \
  --cls-head --epochs 2 --samples-per-epoch 32 --bs 8 --workers 0 --lr 1e-3 --device cpu --out "$W/run"
python scripts/predict_folder.py --ckpt "$W/run/best.pt" --images "$W/syn/val/images" \
  --out "$W/val.npz" --tta 2 --device cpu
python scripts/tune_rules.py --probs "$W/val.npz" --masks "$W/syn/val/masks" --out "$W/rules.json" --top 3
python scripts/visualize.py --probs "$W/val.npz" --images "$W/syn/val/images" --masks "$W/syn/val/masks" \
  --rules "$W/rules.json" --out "$W/vis"
python scripts/make_pseudo_labels.py --probs "$W/val.npz" --images "$W/syn/val/images" --out "$W/pseudo"

python scripts/build_submission.py --models "$W/run/best.pt" "$W/fake_baseline.pt" --rules "$W/rules.json" \
  --tta 2 --dist "$W" --name sub
# 제출물 리허설: 제출 폴더를 작업 폴더로, 네트워크 없이 실행한다고 가정
(cd "$W/sub" && AIF_INPUT_DIR="$W/syn/eval" AIF_PREDICTION_PATH="$W/out/prediction.csv" \
  jupyter nbconvert --to notebook --execute predict.ipynb --output "$W/executed.ipynb" >/dev/null)
python scripts/validate_submission.py "$W/out/prediction.csv" --patches "$W/syn/eval/patches.csv"

for P in empty full; do
  python scripts/build_submission.py --probe $P --dist "$W" --name "probe_$P" >/dev/null
  (cd "$W/probe_$P" && AIF_INPUT_DIR="$W/syn/eval" AIF_PREDICTION_PATH="$W/out/probe_$P.csv" \
    jupyter nbconvert --to notebook --execute predict.ipynb --output "$W/probe_$P.ipynb" >/dev/null)
  python scripts/validate_submission.py "$W/out/probe_$P.csv" --patches "$W/syn/eval/patches.csv"
done
if python scripts/build_submission.py --dist "$W" --name nomodel >/dev/null 2>&1; then
  test -f submission/assets/model/unet_r18_debris_lite.pt || { echo "가중치 없는데 빌드 성공함"; exit 1; }
fi
echo "SMOKE OK"
