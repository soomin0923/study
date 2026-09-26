"""베이스라인 가중치에서 출발하는 파인튜닝.

  python scripts/train.py \
      --train data/aihub_sat:1.0 data/pseudo:2.0 data/negatives:1.0 \
      --val data/val --baseline assets/model/unet_r18_debris_lite.pt \
      --cls-head --epochs 30 --out runs/r18_cls

- --train 은 '폴더:샘플링가중치' 목록입니다 (폴더 규약은 debris_kit/data.py 참고).
- --baseline 을 주면 구조가 같은 부분(UNet-ResNet18)을 베이스라인 가중치로 초기화합니다.
  다른 인코더(--encoder)를 쓰면 ImageNet 가중치로 시작합니다 (학습 PC 에서는 인터넷 필요).
- 매 epoch 검증셋에서 '대회 산식' 점수(간이 규칙 탐색 포함)를 계산해 최고 모델을 저장합니다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from _common import pick_device
from debris_kit.data import DebrisDataset, list_source, read_mask, read_rgb
from debris_kit.infer import predict_probs
from debris_kit.losses import debris_loss
from debris_kit.metric import patch_shape_f, score_from_stats
from debris_kit.model import DebrisNet, init_from_baseline, save_checkpoint
from debris_kit.postprocess import DEFAULT_RULES, make_mask
from debris_kit import MIN_POS_PX


def quick_val(model, val_items, device):
    """TTA 없이 추론 후 몇 개의 임계값만 훑어 산식 점수를 냅니다 (정밀 탐색은 tune_rules.py)."""
    model.eval()
    imgs = np.stack([read_rgb(it["image"]) for it in val_items])
    gts = [read_mask(it["mask"]) if it["mask"] else np.zeros(imgs.shape[1:3], np.uint8) for it in val_items]
    segs, clss = [], []
    for s in range(0, len(imgs), 32):
        seg, cls = predict_probs([model], imgs[s:s + 32], device, tta=1)
        segs.append(seg)
        clss.append(cls)
    seg = np.concatenate(segs)
    cls = None if clss[0] is None else np.concatenate(clss)
    gt_pos = np.array([g.any() for g in gts])
    maxp = seg.reshape(len(seg), -1).max(1)
    best = (-1, None)
    for thr in (0.3, 0.4, 0.5):
        rules = {**DEFAULT_RULES, "thr_abs": thr}
        ms = [make_mask(p, rules) for p in seg]
        final = np.array([int(m.sum()) for m, _ in ms])
        area = np.array([a for _, a in ms])
        shape = np.array([patch_shape_f(m, g) if g.any() else 0.0 for (m, _), g in zip(ms, gts)])
        for t_max in np.arange(0.2, 0.91, 0.1):
            gates = [(t_max, None)] + ([(9.0, t) for t in np.arange(0.2, 0.81, 0.1)] if cls is not None else [])
            for tm, tc in gates:
                pos = (maxp >= tm) & (area >= MIN_POS_PX)
                if tc is not None:
                    pos |= cls >= tc
                pos &= final >= MIN_POS_PX
                r = score_from_stats(gt_pos, pos, shape)
                if r["score"] > best[0]:
                    best = (r["score"], {**r, "thr": thr, "t_max": round(float(tm), 2),
                                        "t_cls": None if tc is None else round(float(tc), 2)})
    model.train()
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="폴더:가중치")
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
    ap.add_argument("--w-f", type=float, default=1.0, help="허용오차 F 손실 가중치")
    ap.add_argument("--w-cls", type=float, default=0.5)
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
        weights += [float(w) / len(src)] * len(src)   # 소스별 총 샘플링 비중 = 가중치
        print(f"train source {path}: {len(src)}개, 가중치 {w}")
    val_items = list_source(a.val)

    arch = {"name": a.arch, "encoder": a.encoder, "cls_head": a.cls_head, "input_scale": a.input_scale}
    use_baseline = a.baseline and a.arch == "Unet" and a.encoder == "resnet18"
    model = DebrisNet(arch, encoder_weights=None if use_baseline else "imagenet")
    if use_baseline:
        new_keys = init_from_baseline(model, a.baseline)
        print("베이스라인에서 초기화, 새로 학습하는 키:", len(new_keys))
    model.to(dev).train()

    ds = DebrisDataset(items, train=True, color_strength=a.color_strength, seed=a.seed)
    sampler = WeightedRandomSampler(weights, a.samples_per_epoch, replacement=True)
    dl = DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    # 베이스라인에서 온 인코더는 작은 학습률, 새 헤드/디코더는 큰 학습률
    enc = [p for n, p in model.named_parameters() if ".encoder." in n]
    rest = [p for n, p in model.named_parameters() if ".encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr * 0.3}, {"params": rest, "lr": a.lr}], weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")

    r0 = quick_val(model, val_items, dev)
    print(f"[epoch 0] val {r0['score']:.4f} f1 {r0['f1']:.3f} shape {r0['shape']:.3f}")
    best, log = r0["score"], []
    save_checkpoint(model, out / "best.pt", extra={"val": r0["score"]})
    for ep in range(1, a.epochs + 1):
        t0, agg = time.time(), {}
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                seg, cls = model(x)
            loss, parts = debris_loss(seg.float(), None if cls is None else cls.float(), y, a.w_f, a.w_cls)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                agg[k] = agg.get(k, 0) + v / len(dl)
        r = quick_val(model, val_items, dev)
        log.append({"epoch": ep, **agg, **{k: r[k] for k in ("score", "f1", "shape")}})
        flag = ""
        if r["score"] > best:
            best, flag = r["score"], " *best"
            save_checkpoint(model, out / "best.pt", extra={"val": best})
        save_checkpoint(model, out / "last.pt")
        loss_s = " ".join(f"{k} {v:.3f}" for k, v in agg.items())
        print(f"[epoch {ep}] {loss_s} | val {r['score']:.4f} f1 {r['f1']:.3f} shape {r['shape']:.3f} "
              f"(thr {r['thr']}, t_max {r['t_max']:.1f}, t_cls {r['t_cls']}) {time.time() - t0:.0f}s{flag}", flush=True)
        (out / "log.json").write_text(json.dumps(log, indent=1))
    print("best val", best)


if __name__ == "__main__":
    main()
