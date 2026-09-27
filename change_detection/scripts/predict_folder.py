"""쌍 폴더에 모델(앙상블)을 돌려 클래스별 확률 지도를 npz 로 저장합니다.
규칙 탐색(tune_rules.py), 의사 라벨(make_pseudo_labels.py), 오류 분석(visualize.py)의 입력입니다.

  python scripts/predict_folder.py --ckpt submission/assets/model/unet_r18_cd.pt \
      --pairs data/val --out runs/val_base.npz --tta 8

--pairs 는 <id>/pre.png, <id>/post.png 가 있는 폴더 (대회 입력 형식이면 images/ 폴더를 주면 됩니다).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from _common import pick_device
from cdkit.data import read_rgb
from cdkit.infer import predict_probs
from cdkit.model import load_checkpoint
from cdkit.postprocess import valid_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = pick_device(a.device)
    models = [load_checkpoint(c, dev) for c in a.ckpt]
    dirs = sorted(d for d in Path(a.pairs).iterdir() if (d / "pre.png").exists())
    ids, segs, clss, valids = [], [], [], []
    t0 = time.time()
    for s in range(0, len(dirs), a.batch):
        chunk = dirs[s:s + a.batch]
        pres = np.stack([read_rgb(d / "pre.png") for d in chunk])
        posts = np.stack([read_rgb(d / "post.png") for d in chunk])
        seg, cls = predict_probs(models, pres, posts, dev, a.tta)
        v = np.stack([valid_mask(p, q) for p, q in zip(pres, posts)])
        ids += [d.name for d in chunk]
        segs.append((seg * v[:, None]).astype(np.float16))
        valids.append(v)
        clss.append(cls if cls is not None else np.full((len(chunk), 2), np.nan, np.float32))
        print(f"{min(s + a.batch, len(dirs))}/{len(dirs)} {time.time() - t0:.1f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, ids=np.array(ids), seg=np.concatenate(segs), cls=np.concatenate(clss),
                        valid=np.concatenate(valids))
    print("saved", a.out)


if __name__ == "__main__":
    main()
