"""앙상블 + D4 TTA 추론. 전/후 영상에 같은 변환을 적용합니다."""
import numpy as np
import torch

from .model import normalize_pair


def _d4(x, k):
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def _d4_inv(x, k):
    x = torch.rot90(x, -(k % 4), dims=[-2, -1])
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict_probs(models, pres: np.ndarray, posts: np.ndarray, device: str, tta: int = 8):
    """pres, posts: (B,H,W,3) uint8
    -> seg (B,2,H,W) float32 [증축 확률, 벌목 확률], cls (B,2) float32 또는 None"""
    x = normalize_pair(pres, posts).to(device)
    amp = device == "cuda"
    seg_sum = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=device)
    cls_sum, n_cls, n_seg = torch.zeros(x.shape[0], 2, device=device), 0, 0
    for model in models:
        for k in range(max(1, tta)):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                seg, cls = model(_d4(x, k))
            p = torch.softmax(seg.float(), dim=1)[:, 1:3]
            seg_sum += _d4_inv(p, k)
            n_seg += 1
            if cls is not None:
                cls_sum += torch.sigmoid(cls.float())
                n_cls += 1
    return (seg_sum / n_seg).cpu().numpy(), ((cls_sum / n_cls).cpu().numpy() if n_cls else None)
