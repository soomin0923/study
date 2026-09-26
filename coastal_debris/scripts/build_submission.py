"""제출 zip 생성: predict.ipynb + requirements.txt + assets/(가중치, 규칙, 설정, debris_kit 코드).

  python scripts/build_submission.py --models runs/a/best.pt runs/b/best.pt --rules runs/rules.json \
      --tta 8 --out dist/sub_v3
  # 베이스라인 가중치 + 튜닝 규칙만으로 제출 (3단계)
  python scripts/build_submission.py --models assets/model/unet_r18_debris_lite.pt --rules runs/rules.json --out dist/sub_v1
  # 리더보드 탐색용 (모델 불필요)
  python scripts/build_submission.py --probe empty --out dist/probe_empty
  python scripts/build_submission.py --probe full  --out dist/probe_full

만든 뒤에는 반드시 로컬에서 네트워크 없이 실행해 보십시오 (README 의 '로컬 리허설').
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nbformat

from _common import ROOT
from debris_kit.postprocess import DEFAULT_RULES

REQUIREMENTS = """\
# 설치 단계(10분 제한)에서 설치됩니다. 필요한 것만 둡니다.
torch>=2.5
segmentation-models-pytorch>=0.5
numpy
opencv-python-headless
"""

INTRO = """\
# 해안쓰레기 영역검출 제출 노트북

- 입력: `AIF_INPUT_DIR` 의 `patches.csv`(id 열) + `images/<id>.png`
- 출력: `AIF_PREDICTION_PATH` 에 `id,rle` CSV
- 모델, 판정 규칙, 설정, 코드(`debris_kit`)는 모두 `assets/` 에 있고 상대경로로 읽습니다 (추론 단계는 오프라인)
- `!`, `%` 줄은 채점 서버에서 제거되므로 쓰지 않습니다
"""

CELLS = [
    ("setup", """\
import csv, json, os, sys, time, zipfile, tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ASSETS = Path("assets")
sys.path.insert(0, str(ASSETS.resolve()))
from debris_kit import H, W, MIN_POS_PX
from debris_kit.rle import rle_encode, validate_submission
from debris_kit.postprocess import DEFAULT_RULES, decide
from debris_kit.model import load_checkpoint
from debris_kit.infer import predict_probs

INPUT_DIR = Path(os.environ.get("AIF_INPUT_DIR", "./input"))
PREDICTION_PATH = Path(os.environ.get("AIF_PREDICTION_PATH", "./prediction.csv"))
CFG = json.loads((ASSETS / "config.json").read_text(encoding="utf-8"))
RULES = {**DEFAULT_RULES, **json.loads((ASSETS / "rules.json").read_text(encoding="utf-8"))}
print("config", CFG)
print("rules", RULES)
"""),
    ("inputs", """\
# 목록 파일(patches.csv)이 추론 대상의 정본입니다. images/ 폴더를 훑어 대상을 정하지 않습니다.
print("AIF_INPUT_DIR:", INPUT_DIR, sorted(p.name for p in INPUT_DIR.iterdir()) if INPUT_DIR.is_dir() else "(없음)")
found = sorted(INPUT_DIR.rglob("patches.csv")) if INPUT_DIR.is_dir() else []
if not found:
    zips = sorted(INPUT_DIR.rglob("*.zip")) if INPUT_DIR.is_dir() else []
    if len(zips) == 1:
        dst = Path(tempfile.mkdtemp()) / "input"
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dst)
        print("압축된 입력을 풀었습니다:", zips[0].name)
        found = sorted(dst.rglob("patches.csv"))
if not found:
    raise SystemExit(f"patches.csv 를 찾지 못했습니다: {INPUT_DIR}")
ROOT = found[0].parent
with found[0].open(encoding="utf-8-sig", newline="") as f:
    IDS = [r["id"].strip() for r in csv.DictReader(f) if r.get("id", "").strip()]
IDS = list(dict.fromkeys(IDS))  # 혹시 모를 중복 id 제거 (중복 행은 채점 실패)
print(f"입력 루트 {ROOT}, 패치 {len(IDS)}건")
"""),
    ("model", """\
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
PROBE = CFG.get("probe")
TTA = int(CFG.get("tta", 8))
MODELS, DEVICE = [], "cpu"
if not PROBE:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        MODELS = [load_checkpoint(ASSETS / m, DEVICE) for m in CFG["models"]]
        predict_probs(MODELS, np.zeros((1, H, W, 3), np.uint8), DEVICE, tta=1)
    except Exception as e:  # noqa: BLE001
        print(f"{DEVICE} 시험 추론 실패({type(e).__name__}: {e}) -> CPU 로 전환")
        DEVICE = "cpu"
        MODELS = [load_checkpoint(ASSETS / m, DEVICE) for m in CFG["models"]]
    if DEVICE == "cpu":
        TTA = min(TTA, int(CFG.get("cpu_tta", 2)))
    print("device", DEVICE, "models", len(MODELS), "tta", TTA)
"""),
    ("infer", """\
def read_image(path):
    buf = np.frombuffer(path.read_bytes(), np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise ValueError(f"{path.name}: 읽기 실패 또는 크기 불일치")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

BATCH = int(CFG.get("batch", 32))
BUDGET = float(CFG.get("time_budget_min", 70)) * 60   # 설치+추론+채점 공용 제한(120분) 대비 여유
ROWS, t0, n_fail = [], time.time(), 0
FULL = f"0 {H * W}"
for s in range(0, len(IDS), BATCH):
    ids = IDS[s:s + BATCH]
    if PROBE == "empty":
        ROWS += [(i, "") for i in ids]; continue
    if PROBE == "full":
        ROWS += [(i, FULL) for i in ids]; continue
    imgs, ok = [], []
    for i in ids:
        try:
            imgs.append(read_image(ROOT / "images" / f"{i}.png")); ok.append(True)
        except Exception as e:  # noqa: BLE001  읽지 못한 패치는 빈 예측으로 내고 계속 진행
            print("읽기 실패:", i, e); imgs.append(np.zeros((H, W, 3), np.uint8)); ok.append(False); n_fail += 1
    seg, cls = predict_probs(MODELS, np.stack(imgs), DEVICE, TTA)
    for k, i in enumerate(ids):
        m = decide(seg[k], None if cls is None else float(cls[k]), RULES) if ok[k] else None
        ROWS.append((i, rle_encode(m) if m is not None and m.sum() >= MIN_POS_PX else ""))
    # 시간 관리: 남은 예상 시간이 예산을 넘으면 TTA 를 줄입니다
    done, el = s + len(ids), time.time() - t0
    if TTA > 1 and done < len(IDS) and el / done * (len(IDS) - done) > BUDGET - el:
        TTA = max(1, TTA // 2); print(f"시간 예산 초과 예상 -> TTA {TTA} 로 축소")
    if (s // BATCH) % 20 == 0 or done >= len(IDS):
        print(f"  {done}/{len(IDS)}  {el:.1f}s  tta {TTA}", flush=True)
n_pos = sum(1 for _, r in ROWS if r)
print(f"추론 완료: {len(ROWS)}건, 양성 {n_pos}건, 읽기 실패 {n_fail}건, {time.time() - t0:.1f}s")
"""),
    ("save", """\
PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
with PREDICTION_PATH.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "rle"])
    w.writerows(ROWS)
print("저장:", PREDICTION_PATH, PREDICTION_PATH.stat().st_size, "bytes")
print("자체 검증:", validate_submission(PREDICTION_PATH, found[0]))
"""),
]


def make_notebook() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO)]
    nb.cells += [nbformat.v4.new_code_cell(f"# [{name}]\n{src}".rstrip()) for name, src in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--rules", default=None, help="tune_rules.py 결과 json (없으면 기본 규칙)")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--cpu-tta", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--time-budget-min", type=float, default=70)
    ap.add_argument("--probe", choices=["empty", "full"], default=None)
    ap.add_argument("--out", required=True, help="제출 폴더 (같은 이름의 .zip 도 생성)")
    a = ap.parse_args()
    if not a.probe and not a.models:
        ap.error("--models 가 필요합니다 (--probe 제외)")

    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets" / "model").mkdir(parents=True)
    nbformat.write(make_notebook(), out / "predict.ipynb")
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    shutil.copytree(ROOT / "debris_kit", out / "assets" / "debris_kit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    names = []
    for k, m in enumerate(a.models):
        name = f"model/m{k}_{Path(m).parent.name}_{Path(m).name}"
        shutil.copy(m, out / "assets" / name)
        names.append(name)
    rules = {**DEFAULT_RULES, **(json.loads(Path(a.rules).read_text()) if a.rules else {})}
    (out / "assets" / "rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    cfg = {"models": names, "tta": a.tta, "cpu_tta": a.cpu_tta, "batch": a.batch,
           "time_budget_min": a.time_budget_min, "probe": a.probe}
    (out / "assets" / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    size = zpath.stat().st_size / 2**20
    print(f"제출 폴더 {out}\nzip {zpath} ({size:.1f} MB, 제한 6GB)")
    print("주의: 참여키가 들어간 파일을 이 폴더에 넣지 마십시오.")


if __name__ == "__main__":
    main()
