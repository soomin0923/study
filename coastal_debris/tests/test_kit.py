"""python -m pytest tests  (또는 python tests/test_kit.py)"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "assets"))

from debris_kit.losses import tolerant_f_loss  # noqa: E402
from debris_kit.metric import (competition_score, mean_shape_from_all_full, patch_shape_f,  # noqa: E402
                               pos_ratio_from_all_empty)
from debris_kit.model import DebrisNet, init_from_baseline, load_checkpoint, save_checkpoint  # noqa: E402
from debris_kit.postprocess import DEFAULT_RULES, decide, make_mask  # noqa: E402
from debris_kit.rle import rle_decode, rle_encode  # noqa: E402


def blob(y0, y1, x0, x1):
    m = np.zeros((256, 256), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_rle_spec_examples():
    m = np.zeros((256, 256), np.uint8)
    m[10:13, 10:15] = 1
    assert rle_encode(m) == "2570 5 2826 5 3082 5"
    last = np.zeros((256, 256), np.uint8)
    last[-1, -1] = 1
    assert rle_encode(last) == "65535 1"
    assert rle_encode(np.zeros((256, 256))) == ""


def test_rle_roundtrip_and_invalid():
    rng = np.random.default_rng(0)
    m = (rng.random((256, 256)) > 0.7).astype(np.uint8)
    assert (rle_decode(rle_encode(m)) == m).all()
    for bad in ["1", "1.5 2", "0 0", "65535 2", "-1 3"]:
        with pytest.raises(ValueError):
            rle_decode(bad)


def test_metric_basic():
    g1, g2 = blob(10, 30, 10, 30), np.zeros((256, 256), np.uint8)
    assert competition_score([g1, g2], [g1, g2])["score"] == pytest.approx(1.0)
    # 1화소 어긋남은 허용
    assert patch_shape_f(blob(11, 31, 11, 31), g1) == pytest.approx(1.0)
    # 50화소 미만 예측은 음성 -> 양성 패치 형상 0, F1 감점
    r = competition_score([blob(10, 15, 10, 15), g2], [g1, g2])
    assert r["shape"] == 0 and r["fn"] == 1


def test_probe_formulas_match_metric():
    rng = np.random.default_rng(1)
    gts = []
    for k in range(40):
        if k % 3 == 0:
            y, x = rng.integers(0, 200, 2)
            gts.append(blob(y, y + int(rng.integers(5, 50)), x, x + int(rng.integers(5, 50))))
        else:
            gts.append(np.zeros((256, 256), np.uint8))
    p_true = np.mean([g.any() for g in gts])
    s0 = competition_score([np.zeros((256, 256))] * 40, gts)["score"]
    assert pos_ratio_from_all_empty(s0) == pytest.approx(p_true)
    full = np.ones((256, 256), np.uint8)
    s1 = competition_score([full] * 40, gts)["score"]
    true_shape = np.mean([patch_shape_f(full, g) for g in gts if g.any()])
    assert mean_shape_from_all_full(s1, p_true) == pytest.approx(true_shape)


def test_guarantee_grows_to_50px():
    prob = np.zeros((256, 256), np.float32)
    prob[100:104, 100:104] = 0.9          # 16 화소만 확신
    prob[90:115, 90:115] += 0.2
    rules = {**DEFAULT_RULES, "guarantee_px": 60, "min_area": 0, "t_max": 0.5}
    m = decide(prob, None, rules)
    assert 60 <= m.sum() <= 80 and m[100:104, 100:104].all()
    assert decide(prob, None, {**rules, "guarantee_px": 0}).sum() == 0  # 보장 없으면 16화소 -> 빈 예측


def test_cls_gate():
    prob = np.full((256, 256), 0.1, np.float32)
    prob[50:60, 50:60] = 0.35
    rules = {**DEFAULT_RULES, "t_cls": 0.6, "thr_rel": 0.9}
    assert decide(prob, 0.2, rules).sum() == 0
    assert decide(prob, 0.8, rules).sum() >= 50


def test_tolerant_loss_matches_metric_on_binary():
    g = blob(20, 60, 20, 40)
    for p in [blob(21, 61, 21, 41), blob(20, 40, 20, 40), blob(100, 120, 100, 120)]:
        lf = tolerant_f_loss(torch.tensor(p, dtype=torch.float32)[None, None],
                             torch.tensor(g, dtype=torch.float32)[None, None]).item()
        assert 1 - lf == pytest.approx(patch_shape_f(p, g), abs=1e-4)


def test_checkpoint_roundtrip_and_baseline_init(tmp_path):
    base = DebrisNet({})  # 베이스라인 구조 (UNet-R18, 헤드 없음)
    torch.save({"state_dict": base.net.state_dict()}, tmp_path / "base.pt")  # 주최측 형식 (arch 없음)
    m = load_checkpoint(tmp_path / "base.pt")
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        assert torch.allclose(m(x)[0], base.eval()(x)[0], atol=1e-5)
    withcls = DebrisNet({"cls_head": True})
    new = init_from_baseline(withcls, tmp_path / "base.pt")
    assert new and all("classification_head" in k for k in new)
    save_checkpoint(withcls, tmp_path / "c.pt")
    seg, cls = load_checkpoint(tmp_path / "c.pt")(x)
    assert seg.shape == (2, 2, 256, 256) and cls.shape == (2,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
