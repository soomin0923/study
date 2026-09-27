"""모델 정의와 체크포인트 입출력.

입력: 전/후 RGB 를 이어 붙인 6채널 (베이스라인과 같음, 각각 ImageNet 정규화)
출력: 3클래스 softmax (0 배경, 1 증축, 2 벌목) + (선택) 분류 헤드 로짓 2개 [증축 있음, 벌목 있음]

체크포인트 형식: {"arch": {...}, "state_dict": {...}}
arch 가 없는 체크포인트(주최 측 unet_r18_cd.pt)는 UNet-ResNet18, 6채널, 3클래스, 분류 헤드 없음으로 읽습니다.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

BASELINE_ARCH = {"name": "Unet", "encoder": "resnet18", "cls_head": False, "input_scale": 1}


class CDNet(nn.Module):
    def __init__(self, arch: dict, encoder_weights: str | None = None):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.arch = {**BASELINE_ARCH, **arch}
        aux = dict(classes=2, pooling="avg", dropout=0.2) if self.arch["cls_head"] else None
        cls = getattr(smp, self.arch["name"])
        self.net = cls(encoder_name=self.arch["encoder"], encoder_weights=encoder_weights,
                       in_channels=6, classes=3, aux_params=aux)
        self.scale = int(self.arch.get("input_scale", 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = self.net(x)
        seg, cls = out if isinstance(out, tuple) else (out, None)
        if self.scale != 1:
            seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
        return seg, cls


def normalize_pair(pre: np.ndarray, post: np.ndarray) -> torch.Tensor:
    """(B,H,W,3) uint8 두 개 -> (B,6,H,W) float32."""
    a = (pre.astype(np.float32) / 255.0 - MEAN) / STD
    b = (post.astype(np.float32) / 255.0 - MEAN) / STD
    x = np.concatenate([a, b], axis=3).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(x))


def load_checkpoint(path, device="cpu") -> CDNet:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = CDNet(ck.get("arch", BASELINE_ARCH))
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
    model.net.load_state_dict(sd)
    return model.to(device).eval()


def init_from_baseline(model: CDNet, baseline_path) -> list[str]:
    """구조가 같은 부분만 베이스라인 가중치로 초기화. 새로 학습할 키 목록을 돌려줍니다."""
    ck = torch.load(Path(baseline_path), map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    own = model.net.state_dict()
    usable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing, _ = model.net.load_state_dict(usable, strict=False)
    return list(missing)


def save_checkpoint(model: CDNet, path, half=True, extra: dict | None = None):
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in model.net.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    extra = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in (extra or {}).items()}
    torch.save({"arch": model.arch, "state_dict": sd, **extra}, path)
