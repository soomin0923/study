"""베이스라인 가중치(unet_r18_cd.pt)에서 출발하는 파인튜닝.

  python scripts/train.py --train data/aihub_cd:1 data/pseudo:2 --val data/val \
      --baseline submission/assets/model/unet_r18_cd.pt --cls-head --epochs 30 --out runs/r18_cls

- --train 은 '폴더:샘플링가중치' 목록 (폴더 규약은 cdkit/data.py)
- 매 epoch 검증셋에서 대회 산식 점수(클래스별 간이 규칙 탐색 포함)를 계산해 최고 모델을 저장합니다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from _common import pick_device
from cdkit import CLASSES, MIN_POS_AREA
from cdkit.data import CDDataset, list_source, load_target, read_rgb
from cdkit.infer import predict_probs
from cdkit.losses import cd_loss
from cdkit.metric import class_score, shape_f_mask
from cdkit.model import CDNet, init_from_baseline, save_checkpoint
from cdkit.postprocess import DEFAULT_CLASS_RULES, make_mask, valid_mask


def quick_val(model, items, device):
    model.eval()
    pres = np.stack([read_rgb(it["pre"]) for it in items])
    posts = np.stack([read_rgb(it["post"]) for it in items])
    tgts = [load_target(it) for it in items]
    segs, clss = [], []
    for s in range(0, len(items), 16):
        seg, cls = predict_probs([model], pres[s:s + 16], posts[s:s + 16], device, tta=1)
        segs.append(seg)
        clss.append(cls)
    seg = np.concatenate(segs)
    cls = None if clss[0] is None else np.concatenate(clss)
    valid = [valid_mask(p, q) for p, q in zip(pres, posts)]
    res = {}
    for k, c in enumerate(CLASSES):
        gts = [(t == k + 1) for t in tgts]
        gt_pos = np.array([g.any() for g in gts])
        prob = seg[:, k] * np.stack(valid)
        maxp = prob.reshape(len(prob), -1).max(1)
        best = None
        for thr in (0.3, 0.4, 0.5):
            r = {**DEFAULT_CLASS_RULES, "thr_abs": thr}
            ms = [make_mask(p, r, v) for p, v in zip(prob, valid)]
            area = np.array([a for _, a in ms])
            final = np.array([int(m.sum()) for m, _ in ms])
            shape = np.array([shape_f_mask(m, g) if g.any() else 0.0 for (m, _), g in zip(ms, gts)])
            gates = [(t, None) for t in np.arange(0.2, 0.91, 0.1)]
            if cls is not None:
                gates += [(9.0, t) for t in np.arange(0.2, 0.81, 0.1)]
            for tm, tc in gates:
                pos = (maxp >= tm) & (area >= MIN_POS_AREA)
                if tc is not None:
                    pos |= cls[:, k] >= tc
                pos &= final >= MIN_POS_AREA
                sc = class_score(gt_pos, pos, shape)
                if best is None or sc["score"] > best["score"]:
                    best = {**sc, "thr": thr, "t_max": round(float(tm), 2),
                            "t_cls": None if tc is None else round(float(tc), 2)}
        res[c] = best
    res["score"] = 0.5 * res[CLASSES[0]]["score"] + 0.5 * res[CLASSES[1]]["score"]
    model.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--cls-head", action="store_true")
    ap.add_argument("--input-scale", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w-f", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=0.5)
    ap.add_argument("--p-same", type=float, default=0.1, help="같은 영상 쌍(변화 없음) 비율")
    ap.add_argument("--color-strength", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = pick_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, weights = [], []
    for spec in a.train:
        path, _, w = spec.rpartition(":") if ":" in spec else (spec, "", "1")
        src = list_source(path)
        items += src
        weights += [float(w) / len(src)] * len(src)
        print(f"train source {path}: {len(src)}쌍, 가중치 {w}")
    val_items = list_source(a.val)

    arch = {"name": a.arch, "encoder": a.encoder, "cls_head": a.cls_head, "input_scale": a.input_scale}
    use_base = a.baseline and a.arch == "Unet" and a.encoder == "resnet18"
    model = CDNet(arch, encoder_weights=None if use_base else "imagenet")
    if use_base:
        print("베이스라인에서 초기화, 새로 학습하는 키:", len(init_from_baseline(model, a.baseline)))
    model.to(dev).train()

    ds = CDDataset(items, train=True, color_strength=a.color_strength, p_same=a.p_same, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True,
                    sampler=WeightedRandomSampler(weights, a.samples_per_epoch, replacement=True),
                    persistent_workers=a.workers > 0)
    enc = [p for n, p in model.named_parameters() if ".encoder." in n]
    rest = [p for n, p in model.named_parameters() if ".encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.3}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=a.epochs * len(dl),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    def fmt(r):
        return " | ".join(f"{c[:4]} {r[c]['score']:.3f}(f1 {r[c]['f1']:.2f} sh {r[c]['shape']:.2f})" for c in CLASSES)

    r0 = quick_val(model, val_items, dev)
    print(f"[epoch 0] val {r0['score']:.4f}  {fmt(r0)}")
    best, log = r0["score"], []
    save_checkpoint(model, out / "best.pt", extra={"val": best})
    for ep in range(1, a.epochs + 1):
        t0, agg = time.time(), {}
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                seg, cls = model(x)
            loss, parts = cd_loss(seg.float(), None if cls is None else cls.float(), y, a.w_f, a.w_cls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        r = quick_val(model, val_items, dev)
        log.append({"epoch": ep, **agg, "score": r["score"], **{c: r[c]["score"] for c in CLASSES}})
        flag = ""
        if r["score"] > best:
            best, flag = r["score"], " *best"
            save_checkpoint(model, out / "best.pt", extra={"val": best})
        save_checkpoint(model, out / "last.pt")
        print(f"[epoch {ep}] " + " ".join(f"{k} {v:.3f}" for k, v in agg.items())
              + f" | val {r['score']:.4f}  {fmt(r)}  {time.time() - t0:.0f}s{flag}", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    print("best val", best)


if __name__ == "__main__":
    main()
