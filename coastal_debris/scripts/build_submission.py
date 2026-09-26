"""submission/ 템플릿으로 제출 zip 을 만듭니다.

submission/ 폴더가 곧 zip 의 내용입니다 (zip 최상위 = 이 폴더 안):
  predict.ipynb        진입점
  requirements.txt     설치 패키지
  assets/              debris_kit 코드, config.json(모델 목록·TTA), rules.json(판정 규칙), model/(가중치)

  # 템플릿 그대로 (submission/assets/model/unet_r18_debris_lite.pt 를 넣어 둔 경우)
  python scripts/build_submission.py
  # 모델·규칙 지정
  python scripts/build_submission.py --models runs/a/best.pt runs/b/best.pt --rules runs/rules.json --name sub_v2
  # 리더보드 탐색용 (모델 불필요)
  python scripts/build_submission.py --probe empty --name probe_empty

결과: dist/<name>/ (펼친 폴더) 와 dist/<name>.zip. 템플릿(submission/)은 바꾸지 않습니다.
"""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

from _common import ROOT
from debris_kit.postprocess import DEFAULT_RULES

TEMPLATE = ROOT / "submission"
SKIP = {"__pycache__", ".gitkeep", ".ipynb_checkpoints", ".DS_Store"}


def build(out: Path, models, rules_path, probe, tta, cpu_tta, batch, budget, allow_missing):
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(TEMPLATE, out, ignore=shutil.ignore_patterns(*SKIP, "*.pyc"))
    cfg = json.loads((out / "assets" / "config.json").read_text(encoding="utf-8"))

    if models:  # 지정한 모델로 교체 (템플릿에 들어 있던 가중치는 뺍니다)
        shutil.rmtree(out / "assets" / "model")
        (out / "assets" / "model").mkdir()
        cfg["models"] = []
        for k, m in enumerate(models):
            name = f"model/m{k}_{Path(m).parent.name}_{Path(m).name}"
            shutil.copy(m, out / "assets" / name)
            cfg["models"].append(name)
    if probe:
        shutil.rmtree(out / "assets" / "model")
        (out / "assets" / "model").mkdir()
        cfg["models"] = []
    cfg.update({"probe": probe, "tta": tta, "cpu_tta": cpu_tta, "batch": batch, "time_budget_min": budget})
    (out / "assets" / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if rules_path:
        rules = {**DEFAULT_RULES, **json.loads(Path(rules_path).read_text())}
        (out / "assets" / "rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    missing = [m for m in cfg["models"] if not (out / "assets" / m).is_file()]
    if missing:
        msg = f"가중치 파일 없음: {missing} -> {TEMPLATE / 'assets'} 아래에 넣거나 --models 로 지정하십시오"
        if not allow_missing:
            sys.exit(msg)
        print("경고:", msg, "(이 zip 은 그대로 제출하면 실패합니다)")

    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out).as_posix())
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
    assert "predict.ipynb" in names and "requirements.txt" in names, "zip 최상위 구성 오류"
    return zpath, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--rules", default=None, help="tune_rules.py 결과 json (없으면 템플릿의 rules.json)")
    ap.add_argument("--probe", choices=["empty", "full"], default=None)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--cpu-tta", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--time-budget-min", type=float, default=70)
    ap.add_argument("--name", default=None, help="출력 이름 (기본: submission 또는 probe_<종류>)")
    ap.add_argument("--dist", default=str(ROOT / "dist"))
    ap.add_argument("--allow-missing-weights", action="store_true", help="가중치 없이 구조만 묶기")
    a = ap.parse_args()

    name = a.name or (f"probe_{a.probe}" if a.probe else "submission")
    zpath, names = build(Path(a.dist) / name, a.models, a.rules, a.probe, a.tta, a.cpu_tta, a.batch,
                         a.time_budget_min, a.allow_missing_weights)
    print(f"zip: {zpath} ({zpath.stat().st_size / 2**20:.1f} MB, 제한 6GB)")
    for n in names:
        if not n.startswith("assets/debris_kit/"):
            print("  ", n)
    print("   assets/debris_kit/*.py", sum(n.startswith("assets/debris_kit/") for n in names), "개")
    print("주의: 참여키가 들어간 파일을 제출물에 넣지 마십시오.")


if __name__ == "__main__":
    main()
