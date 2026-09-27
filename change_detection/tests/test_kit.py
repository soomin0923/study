"""python -m pytest tests -q"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submission" / "assets"))

from cdkit.geom import CellError, mask_to_polygons, parse_cell, polygons_to_cell, polygons_to_geometry, rasterize  # noqa: E402
from cdkit.losses import tolerant_f_loss  # noqa: E402
from cdkit.metric import mean_neg_f1_from_all_empty, score_geoms, shape_f_mask, shape_f_poly  # noqa: E402
from cdkit.model import CDNet, init_from_baseline, load_checkpoint, save_checkpoint  # noqa: E402
from cdkit.postprocess import DEFAULT_CLASS_RULES, decide, decide_class, valid_mask  # noqa: E402
from cdkit.submission import read_submission  # noqa: E402


def rect(y0, y1, x0, x1):
    m = np.zeros((256, 256), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_spec_examples_parse():
    a = parse_cell("[[[10,10],[30,10],[30,22],[10,22]]]")
    assert polygons_to_geometry(a).area == pytest.approx(240)
    b = parse_cell("[[[128.01,54.11],[153.99,69.11],[141.99,89.89],[116.01,74.89]]]")
    assert polygons_to_geometry(b).area == pytest.approx(720, rel=0.001)  # 30도 회전한 30x24 사각형
    assert parse_cell("") == [] and parse_cell("[]") == []


@pytest.mark.parametrize("bad", [
    "not json", "{}", "[[10,10],[30,10],[30,22]]",                  # 객체, 점 목록만
    "[[[[10,10],[30,10],[30,22],[10,22]]]]",                         # 한 겹 더
    '[[[10,"a"],[30,10],[30,22]]]', "[[[10,NaN],[30,10],[30,22]]]",  # 숫자 아님, NaN
])
def test_invalid_cells(bad):
    with pytest.raises(CellError):
        parse_cell(bad)


def test_geometry_rules():
    assert polygons_to_geometry([[[0, 0], [1, 1]]]).is_empty                 # 3점 미만 무시
    g = polygons_to_geometry([[[-10, -10], [20, -10], [20, 20], [-10, 20]]])  # 패치 밖 잘라냄
    assert g.area == pytest.approx(400)


def test_mask_polygon_roundtrip():
    m = rect(10, 30, 40, 70) | rect(100, 140, 100, 120)
    polys = mask_to_polygons(m, simplify_px=0)
    assert polygons_to_geometry(polys).area == pytest.approx(m.sum())
    assert (rasterize(polys) == m).all()
    json.loads(polygons_to_cell(polys))


def test_metric_exact_and_raster():
    g = rect(20, 40, 20, 60)
    p = rect(21, 41, 21, 61)                                   # 1화소 어긋남은 허용
    gg = polygons_to_geometry(mask_to_polygons(g, 0))
    pg = polygons_to_geometry(mask_to_polygons(p, 0))
    assert shape_f_poly(pg, gg) == pytest.approx(1.0)
    assert shape_f_mask(p, g) == pytest.approx(1.0)
    far = rect(120, 140, 120, 160)
    assert shape_f_mask(far, g) == 0
    ids = ["a", "b"]
    empty = polygons_to_geometry([])
    geo = {"new_building": {"a": gg, "b": empty}, "tree_removal": {"a": empty, "b": gg}}
    assert score_geoms(geo, geo, ids)["score"] == pytest.approx(1.0)


def test_probe_formula():
    n, p = mean_neg_f1_from_all_empty(0.125 * (2 * (1 - 0.3) / (2 - 0.3)) * 2)
    assert p == pytest.approx(0.3)


def test_valid_mask_and_decide():
    pre = np.full((256, 256, 3), 120, np.uint8)
    post = pre.copy()
    post[:, :100] = 0                                          # 무영상
    pre[50, 200] = 0                                           # 그림자 속 점 하나는 무영상 아님
    v = valid_mask(pre, post)
    assert not v[:, :100].any() and v[50, 200] and v[:, 110:].all()
    prob = np.zeros((2, 256, 256), np.float32)
    prob[0, 40:60, 60:140] = 0.9                               # 무영상에 걸친 증축 예측
    out = decide(prob, None, {}, v)
    g = polygons_to_geometry(out["new_building"])
    assert g.area > 20 and g.bounds[0] >= 100
    assert out["tree_removal"] == []


def test_guarantee_min_area():
    prob = np.zeros((256, 256), np.float32)
    prob[100:103, 100:103] = 0.9                               # 9화소
    prob[95:110, 95:110] += 0.2
    r = {**DEFAULT_CLASS_RULES, "min_area": 0, "min_cc": 0}
    assert decide_class(prob, None, r).sum() >= 20
    assert decide_class(prob, None, {**r, "guarantee_px": 0}).sum() == 0


def test_tolerant_loss_matches_raster_metric():
    g = rect(20, 60, 20, 40)
    for p in (rect(21, 61, 21, 41), rect(20, 40, 20, 40), rect(100, 120, 100, 120)):
        lf = tolerant_f_loss(torch.tensor(p, dtype=torch.float32)[None, None],
                             torch.tensor(g, dtype=torch.float32)[None, None]).item()
        assert 1 - lf == pytest.approx(shape_f_mask(p, g), abs=1e-4)


def test_checkpoint_baseline_format(tmp_path):
    base = CDNet({})
    torch.save({"state_dict": base.net.state_dict()}, tmp_path / "base.pt")   # 주최측 형식 (arch 없음)
    m = load_checkpoint(tmp_path / "base.pt")
    x = torch.randn(2, 6, 256, 256)
    with torch.no_grad():
        assert torch.allclose(m(x)[0], base.eval()(x)[0], atol=1e-5)
    c = CDNet({"cls_head": True})
    new = init_from_baseline(c, tmp_path / "base.pt")
    assert new and all("classification_head" in k for k in new)
    save_checkpoint(c, tmp_path / "c.pt")
    seg, cls = load_checkpoint(tmp_path / "c.pt")(x)
    assert seg.shape == (2, 3, 256, 256) and cls.shape == (2, 2)


def _write(path, rows, header=("id", "new_building", "tree_removal")):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_submission_rules(tmp_path):
    p = tmp_path / "p.csv"
    _write(p, [("u1", "[[[10,10],[30,10],[30,22],[10,22]]]", ""), ("u2", "", "")])
    g = read_submission(p, ["u1", "u2", "u3"])
    assert g["new_building"]["u1"].area == pytest.approx(240)
    for rows, header in [([("u1", "", ""), ("u1", "", "")], None), ([("zz", "", "")], None),
                         ([("u1", "")], ("id", "new_building"))]:
        _write(p, rows, header or ("id", "new_building", "tree_removal"))
        with pytest.raises(ValueError):
            read_submission(p, ["u1", "u2"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
