"""앙상블 + D4 TTA 추론. 결과는 패치별 debris 확률 지도와 분류 확률."""
import numpy as np
import torch

from .model import normalize


def _d4(x, k):
    """k: 0..7. rot90 k%4 회, k>=4 이면 좌우반전 추가."""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def _d4_inv(x, k):
    x = torch.rot90(x, -(k % 4), dims=[-2, -1])
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return x


@torch.no_grad()
def predict_probs(models, imgs: np.ndarray, device: str, tta: int = 8):
    """imgs: (B,H,W,3) uint8 -> (seg_prob (B,H,W) float32, cls_prob (B,) float32 또는 None).
    분류 헤드가 있는 모델이 하나도 없으면 cls_prob 는 None."""
    x = normalize(imgs).to(device)
    use_amp = device == "cuda"
    seg_sum = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=device)
    cls_sum, n_cls, n_seg = torch.zeros(x.shape[0], device=device), 0, 0
    for model in models:
        for k in range(max(1, tta)):
            xi = _d4(x, k)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                seg, cls = model(xi)
            p = torch.softmax(seg.float(), dim=1)[:, 1:2]
            seg_sum += _d4_inv(p, k)[:, 0]
            n_seg += 1
            if cls is not None:
                cls_sum += torch.sigmoid(cls.float())
                n_cls += 1
    seg_prob = (seg_sum / n_seg).cpu().numpy()
    cls_prob = (cls_sum / n_cls).cpu().numpy() if n_cls else None
    return seg_prob, cls_prob
