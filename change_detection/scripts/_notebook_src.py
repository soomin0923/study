"""submission/predict.ipynb 생성기. 노트북을 고칠 때는 이 파일을 고치고
python scripts/_notebook_src.py 로 다시 만드십시오 (노트북과 이 파일이 어긋나지 않게)."""
from pathlib import Path

import nbformat

INTRO = """\
# 불법시설물 변화탐지 제출 노트북 (주제3)

- 입력: `AIF_INPUT_DIR` 의 `pairs.csv`(id 열) + `images/<id>/pre.png`, `images/<id>/post.png`
- 출력: `AIF_PREDICTION_PATH` 에 `id,new_building,tree_removal` CSV (각 칸은 폴리곤 목록 JSON, 없으면 빈 문자열)
- 모델, 판정 규칙, 설정, 코드(`cdkit`)는 모두 `assets/` 에 있고 상대경로로 읽습니다 (추론 단계는 오프라인)
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
from cdkit import H, W, CLASSES
from cdkit.geom import polygons_to_cell
from cdkit.postprocess import DEFAULT_CLASS_RULES, decide, valid_mask
from cdkit.model import load_checkpoint
from cdkit.infer import predict_probs
from cdkit.submission import read_submission

INPUT_DIR = Path(os.environ.get("AIF_INPUT_DIR", "./input"))
PREDICTION_PATH = Path(os.environ.get("AIF_PREDICTION_PATH", "./prediction.csv"))
CFG = json.loads((ASSETS / "config.json").read_text(encoding="utf-8"))
_r = json.loads((ASSETS / "rules.json").read_text(encoding="utf-8"))
RULES = {c: {**DEFAULT_CLASS_RULES, **_r.get(c, {})} for c in CLASSES}
print("config", CFG)
print("rules", json.dumps(RULES, ensure_ascii=False))
"""),
    ("inputs", """\
# 목록 파일(pairs.csv)이 추론 대상의 정본입니다. images/ 폴더를 훑어 대상을 정하지 않습니다.
print("AIF_INPUT_DIR:", INPUT_DIR, sorted(p.name for p in INPUT_DIR.iterdir()) if INPUT_DIR.is_dir() else "(없음)")
found = sorted(INPUT_DIR.rglob("pairs.csv")) if INPUT_DIR.is_dir() else []
if not found:
    zips = sorted(INPUT_DIR.rglob("*.zip")) if INPUT_DIR.is_dir() else []
    if len(zips) == 1:
        dst = Path(tempfile.mkdtemp()) / "input"
        with zipfile.ZipFile(zips[0]) as z:
            z.extractall(dst)
        print("압축된 입력을 풀었습니다:", zips[0].name)
        found = sorted(dst.rglob("pairs.csv"))
if not found:
    raise SystemExit(f"pairs.csv 를 찾지 못했습니다: {INPUT_DIR}")
ROOT = found[0].parent
with found[0].open(encoding="utf-8-sig", newline="") as f:
    IDS = [r["id"].strip() for r in csv.DictReader(f) if r.get("id", "").strip()]
IDS = list(dict.fromkeys(IDS))   # 중복 행은 채점 실패이므로 제거
print(f"입력 루트 {ROOT}, 쌍 {len(IDS)}건")
"""),
    ("model", """\
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available())
PROBE = CFG.get("probe")
TTA = int(CFG.get("tta", 8))
MODELS, DEVICE = [], "cpu"
if not PROBE:
    missing = [m for m in CFG["models"] if not (ASSETS / m).is_file()]
    if missing:
        raise SystemExit(f"assets/ 에 가중치 파일이 없습니다: {missing}")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        MODELS = [load_checkpoint(ASSETS / m, DEVICE) for m in CFG["models"]]
        z = np.zeros((1, H, W, 3), np.uint8)
        predict_probs(MODELS, z, z, DEVICE, tta=1)
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
    im = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None or im.shape[:2] != (H, W):
        raise ValueError(f"{path}: 읽기 실패 또는 크기 불일치")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

BATCH = int(CFG.get("batch", 16))
BUDGET = float(CFG.get("time_budget_min", 70)) * 60
ROWS, t0, n_fail = [], time.time(), 0
for s in range(0, len(IDS), BATCH):
    ids = IDS[s:s + BATCH]
    if PROBE == "empty":
        ROWS += [(i, "", "") for i in ids]; continue
    pres, posts, ok = [], [], []
    for i in ids:
        try:
            a = read_image(ROOT / "images" / i / "pre.png")
            b = read_image(ROOT / "images" / i / "post.png")
            ok.append(True)
        except Exception as e:  # noqa: BLE001  읽지 못한 쌍은 빈 예측으로 내고 계속 진행
            print("읽기 실패:", i, e); n_fail += 1; ok.append(False)
            a = b = np.zeros((H, W, 3), np.uint8)
        pres.append(a); posts.append(b)
    seg, cls = predict_probs(MODELS, np.stack(pres), np.stack(posts), DEVICE, TTA)
    for k, i in enumerate(ids):
        if not ok[k]:
            ROWS.append((i, "", "")); continue
        v = valid_mask(pres[k], posts[k])     # 두 영상 모두 검지 않은 구역만 판정 대상
        out = decide(seg[k], None if cls is None else cls[k], RULES, v)
        ROWS.append((i, *(polygons_to_cell(out[c]) for c in CLASSES)))
    done, el = s + len(ids), time.time() - t0
    if TTA > 1 and done < len(IDS) and el / done * (len(IDS) - done) > BUDGET - el:
        TTA = max(1, TTA // 2); print(f"시간 예산 초과 예상 -> TTA {TTA} 로 축소")
    if (s // BATCH) % 20 == 0 or done >= len(IDS):
        print(f"  {done}/{len(IDS)}  {el:.1f}s  tta {TTA}", flush=True)
n_b = sum(1 for r in ROWS if r[1]); n_t = sum(1 for r in ROWS if r[2])
print(f"추론 완료: {len(ROWS)}건, 증축 양성 {n_b}, 벌목 양성 {n_t}, 읽기 실패 {n_fail}, {time.time() - t0:.1f}s")
"""),
    ("save", """\
PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
with PREDICTION_PATH.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", *CLASSES])
    w.writerows(ROWS)
print("저장:", PREDICTION_PATH, PREDICTION_PATH.stat().st_size, "bytes")
g = read_submission(PREDICTION_PATH, IDS)   # 채점 실패 규칙 자체 검사 (실패하면 여기서 오류)
print("자체 검증 OK:", {c: sum(1 for x in g[c].values() if x.area >= 20) for c in CLASSES})
"""),
]


def make_notebook():
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO)]
    nb.cells += [nbformat.v4.new_code_cell(f"# [{n}]\n{s}".rstrip()) for n, s in CELLS]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "submission" / "predict.ipynb"
    nbformat.write(make_notebook(), out)
    print("wrote", out)
