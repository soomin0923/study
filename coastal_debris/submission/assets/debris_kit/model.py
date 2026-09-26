"""모델 정의와 체크포인트 입출력.

체크포인트 형식: {"arch": {...}, "state_dict": {...}}
- arch 가 없는 체크포인트(주최 측 베이스라인 unet_r18_debris_lite.pt)는 UNet-ResNet18, 분류 헤드 없음으로 읽습니다.
- 분할 출력은 베이스라인과 같은 2클래스(softmax) 입니다. debris 확률 = softmax[:, 1]
- 분류 헤드(선택)는 smp 의 aux_params 로 붙이며 로짓 1개(패치에 쓰레기가 있는가)를 냅니다.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

BASELINE_ARCH = {"name": "Unet", "encoder": "resnet18", "cls_head": False, "input_scale": 1}


class DebrisNet(nn.Module):
    def __init__(self, arch: dict, encoder_weights: str | None = None):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.arch = {**BASELINE_ARCH, **arch}
        aux = dict(classes=1, pooling="avg", dropout=0.2) if self.arch["cls_head"] else None
        cls = getattr(smp, self.arch["name"])
        self.net = cls(encoder_name=self.arch["encoder"], encoder_weights=encoder_weights,
                       in_channels=3, classes=2, aux_params=aux)
        self.scale = int(self.arch.get("input_scale", 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        if self.scale != 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        out = self.net(x)
        seg, cls = (out if isinstance(out, tuple) else (out, None))
        if self.scale != 1:
            seg = F.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
        return seg, (cls.squeeze(1) if cls is not None else None)


def normalize(imgs: np.ndarray) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float32 정규화 텐서."""
    x = (imgs.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))


def load_checkpoint(path, device="cpu") -> DebrisNet:
    ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    arch = ck.get("arch", BASELINE_ARCH)
    model = DebrisNet(arch)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()}
    model.net.load_state_dict(sd)
    return model.to(device).eval()


def init_from_baseline(model: DebrisNet, baseline_path) -> list[str]:
    """베이스라인 가중치로 초기화 (구조가 같은 부분만). 새로 생긴 키 목록을 돌려줍니다."""
    ck = torch.load(Path(baseline_path), map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    own = model.net.state_dict()
    usable = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    missing, _ = model.net.load_state_dict(usable, strict=False)
    return list(missing)


def save_checkpoint(model: DebrisNet, path, half=True, extra: dict | None = None):
    sd = {k: (v.half() if half and v.is_floating_point() else v).cpu() for k, v in model.net.state_dict().items()}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    extra = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in (extra or {}).items()}
    torch.save({"arch": model.arch, "state_dict": sd, **extra}, path)
