"""폴더의 패치에 모델(앙상블)을 돌려 확률 지도를 npz 로 저장합니다.
규칙 탐색(tune_rules.py), 의사 라벨(make_pseudo_labels.py), 오류 분석(visualize.py)의 입력입니다.

  python scripts/predict_folder.py --ckpt assets/model/unet_r18_debris_lite.pt \
      --images data/val/images --out runs/val_baseline.npz --tta 8
"""
import argparse
import time
from pathlib import Path

import numpy as np

from _common import pick_device
from debris_kit.data import read_rgb
from debris_kit.infer import predict_probs
from debris_kit.model import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--images", required=True, help="<id>.png 들이 있는 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", type=int, default=8, help="1=없음, 8=D4 전체")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = pick_device(a.device)
    models = [load_checkpoint(c, dev) for c in a.ckpt]
    paths = sorted(Path(a.images).glob("*.png"))
    ids, segs, clss = [], [], []
    t0 = time.time()
    for s in range(0, len(paths), a.batch):
        chunk = paths[s:s + a.batch]
        imgs = np.stack([read_rgb(p) for p in chunk])
        seg, cls = predict_probs(models, imgs, dev, a.tta)
        ids += [p.stem for p in chunk]
        segs.append(seg.astype(np.float16))
        clss.append(cls if cls is not None else np.full(len(chunk), np.nan, np.float32))
        print(f"{min(s + a.batch, len(paths))}/{len(paths)} {time.time() - t0:.1f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, ids=np.array(ids), seg=np.concatenate(segs), cls=np.concatenate(clss))
    print("saved", a.out)


if __name__ == "__main__":
    main()
