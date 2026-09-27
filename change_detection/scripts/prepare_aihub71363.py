"""AIHub 71363 (국립공원 변화탐지) Skyset 0.5m -> 주제3 학습용 전/후 쌍.

이 데이터에서 확인된 사실 (검증 라벨·JSON 샘플 분석)
- 1024x1024, 0.5m, EPSG:32652. 같은 타일 ID 가 여러 날짜로 촬영됨 (7~12월, 간격 8~100일)
- 라벨 = 날짜별 토지피복 (10 건물, 20 하천, 30 도로, 40 논, 60 산림, 100 비대상지, 그 외 1·50·80 소량)
- 두 날짜 사이 건물·하천·도로·논 라벨은 완전히 같음 (복사됨). 산림 <-> 비대상지만 바뀜 (날짜별 판독 제외 영역)
  => 날짜 간 라벨 차이는 '변화 라벨'로 쓸 수 없음

그래서 이렇게 만듭니다.
1) 같은 타일의 두 날짜 사진 = 실제 전/후 쌍 (계절·그림자·색감·시차 차이가 진짜로 들어 있음). 기본 정답은 '변화 없음'
2) 그 쌍에 변화를 합성
   - 증축: 건물(10) 조각을 '전' 사진에서 지움(inpaint) -> 후 사진에만 있는 건물 = new_building
   - 벌목: 두 날짜 모두 산림(60)인 곳의 덩어리를 '후' 사진에서 맨땅 질감으로 -> tree_removal
3) 비대상지(100)는 학습에서 무시 (ignore.png)
4) 창 크기를 256 x (0.55~0.7)/0.5 로 잘라 256 으로 줄여 대회 축척에 맞춤

  python scripts/prepare_aihub71363.py --images <원천 폴더> --labels <라벨 TIF 폴더> --out data/aihub_cd_val
  (원천이 4밴드면 --bands 로 RGB 순서 지정. 예: SkySat analytic BGRN 이면 --bands 2 1 0)

원천 사진 형식(밴드 수·비트 수)은 샘플로 확인해야 합니다. 첫 타일에서 형식을 출력하니 확인하십시오.
"""
import argparse
import collections
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from _common import ROOT  # noqa: F401
from prepare_aihub import bare_texture, random_blob_in

ID_RE = re.compile(r"(SS\d{10})_(\d{8})")
BUILDING, FOREST, NONTARGET = 10, 60, 100
EXTS = (".tif", ".tiff", ".png", ".jpg")


def index(folder):
    out = collections.defaultdict(dict)
    for p in Path(folder).rglob("*"):
        if p.suffix.lower() in EXTS:
            m = ID_RE.search(p.name)
            if m:
                out[m.group(1)][m.group(2)] = p
    return out


def read_label(path):
    im = Image.open(path)
    lab = np.array(im)
    tie = None
    try:
        t = im.tag_v2.get(33922)            # GeoTIFF tiepoint (…, X, Y, …)
        s = im.tag_v2.get(33550)            # pixel scale
        if t and s:
            tie = (float(t[3]), float(t[4]), float(s[0]))
    except Exception:  # noqa: BLE001
        pass
    return lab, tie


def to_rgb8(arr, bands):
    """원천 -> RGB uint8. 16비트면 영상별 2~98% 로 늘림 (대회 영상도 영상별로 따로 밝기를 편 규격)."""
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    if a.shape[0] <= 8 and a.shape[-1] > 8:  # (C,H,W) -> (H,W,C)
        a = np.moveaxis(a, 0, -1)
    a = a[..., list(bands)]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    a = a.astype(np.float32)
    valid = a.max(-1) > 0
    out = np.zeros_like(a)
    for c in range(3):
        v = a[..., c][valid]
        if v.size == 0:
            continue
        lo, hi = np.percentile(v, [2, 98])
        out[..., c] = (a[..., c] - lo) / max(hi - lo, 1e-6) * 255
    out[~valid] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def read_image(path, bands):
    try:
        import tifffile
        a = tifffile.imread(str(path))
    except Exception:  # noqa: BLE001  tifffile 이 없거나 일반 이미지
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is not None and a.ndim == 3 and a.shape[-1] >= 3:
            a = a[..., [2, 1, 0] + list(range(3, a.shape[-1]))]   # BGR(A) -> RGB(A)
    if a is None:
        raise ValueError(f"읽기 실패: {path}")
    return a


def offset_px(tie_a, tie_b):
    """같은 타일 두 날짜의 좌표 차이를 화소로. (dx, dy) = b 가 a 보다 오른쪽·아래로 몇 화소 밀렸는지."""
    if tie_a is None or tie_b is None:
        return 0, 0
    dx = round((tie_b[0] - tie_a[0]) / tie_a[2])
    dy = round((tie_a[1] - tie_b[1]) / tie_a[2])
    return dx, dy


def to_patch(img, lab_list, size=256):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    labs = [cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) for m in lab_list]
    return img, labs


def save(out, name, pre, post, bld, tree, ign):
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "pre.png"), pre[..., ::-1])
    cv2.imwrite(str(d / "post.png"), post[..., ::-1])
    cv2.imwrite(str(d / "building.png"), bld * 255)
    cv2.imwrite(str(d / "tree.png"), tree * 255)
    if ign.any():
        cv2.imwrite(str(d / "ignore.png"), ign.astype(np.uint8) * 255)


def synth_new_building(pre, post_lab, rng, p):
    """후 날짜 건물 라벨 중 일부를 '전' 사진에서 지워 새 건물로 만듦."""
    b = (post_lab == BUILDING).astype(np.uint8)
    new = np.zeros_like(b)
    n, lab = cv2.connectedComponents(b)
    for k in range(1, n):
        comp = lab == k
        ys, xs = np.nonzero(comp)
        on_edge = ys.min() == 0 or xs.min() == 0 or ys.max() == b.shape[0] - 1 or xs.max() == b.shape[1] - 1
        if not on_edge and rng.random() < p:
            new[comp] = 1
    if new.any():
        hole = cv2.dilate(new, np.ones((7, 7), np.uint8))       # 그림자·벽 여유까지 지움
        pre = cv2.inpaint(pre, hole, 9, cv2.INPAINT_TELEA)
        # inpaint 자리는 매끈해서 '흐린 자국'이 단서가 될 수 있음 -> 주변 질감 크기의 잡음을 입힘
        ring = (cv2.dilate(hole, np.ones((9, 9), np.uint8)) > 0) & (hole == 0)
        if ring.any():
            hi = pre.astype(np.float32) - cv2.GaussianBlur(pre, (0, 0), 1.5).astype(np.float32)
            sd = hi[ring].std(axis=0)
            noise = rng.normal(0, 1, pre.shape).astype(np.float32) * sd
            noise = cv2.GaussianBlur(noise, (0, 0), 0.7)
            m = hole.astype(bool)
            pre = pre.astype(np.float32)
            pre[m] += noise[m]
            pre = np.clip(pre, 0, 255).astype(np.uint8)
    return pre, new


def synth_tree_removal(post, forest_both, rng, p, lo, hi):
    if rng.random() >= p:
        return post, np.zeros(forest_both.shape, np.uint8)
    cut = random_blob_in(forest_both, rng, lo, hi)
    if cut is None:
        return post, np.zeros(forest_both.shape, np.uint8)
    tex = bare_texture(rng, post.shape)
    soft = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 1.2)[..., None]
    post = (post * (1 - soft) + tex * soft).astype(np.uint8)
    return post, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="원천(VS_02/TS_02) 폴더")
    ap.add_argument("--labels", required=True, help="라벨 TIF(VL_01/TL_01) 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bands", type=int, nargs=3, default=[0, 1, 2], help="원천에서 R G B 로 쓸 밴드 번호")
    ap.add_argument("--windows", type=int, default=6, help="쌍마다 뽑을 창 수")
    ap.add_argument("--max-pairs", type=int, default=2, help="타일마다 쓸 날짜 쌍 수")
    ap.add_argument("--single", action="store_true", help="날짜가 하나뿐인 타일도 전=후 로 사용")
    ap.add_argument("--p-build", type=float, default=0.35, help="건물 하나를 새 건물로 만들 확률")
    ap.add_argument("--p-tree", type=float, default=0.35, help="창 하나에 벌목을 합성할 확률")
    ap.add_argument("--max-ignore", type=float, default=0.5, help="비대상지 비율이 이보다 큰 창은 버림")
    ap.add_argument("--building-focus", type=float, default=0.5,
                    help="건물이 있는 타일에서 창을 건물 위치 중심으로 뽑을 확률 (국립공원은 건물이 드묾)")
    ap.add_argument("--limit", type=int, default=0, help="처리할 타일 수 상한 (시험용)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    imgs, labs = index(a.images), index(a.labels)
    tiles = sorted(set(imgs) & set(labs))
    print(f"원천 타일 {len(imgs)}, 라벨 타일 {len(labs)}, 둘 다 있는 타일 {len(tiles)}")
    if a.limit:
        tiles = tiles[:a.limit]
    stat = collections.Counter()
    shown = False
    for t in tiles:
        dates = sorted(set(imgs[t]) & set(labs[t]))
        pairs = list(zip(dates, dates[1:]))[:a.max_pairs]
        if not pairs and a.single and dates:
            pairs = [(dates[0], dates[0])]
        for d1, d2 in pairs:
            try:
                raw1, raw2 = read_image(imgs[t][d1], a.bands), read_image(imgs[t][d2], a.bands)
            except ValueError as e:
                print(e)
                stat["read_fail"] += 1
                continue
            if not shown:
                print(f"원천 형식: {imgs[t][d1].name} shape {raw1.shape} dtype {raw1.dtype} "
                      f"-> RGB 밴드 {a.bands} (다르면 --bands 로 바꾸십시오)")
                shown = True
            i1, i2 = to_rgb8(raw1, a.bands), to_rgb8(raw2, a.bands)
            l1, tie1 = read_label(labs[t][d1])
            l2, tie2 = read_label(labs[t][d2])
            dx, dy = offset_px(tie1, tie2)
            if abs(dx) > 64 or abs(dy) > 64:
                stat["offset_too_big"] += 1
                continue
            # 두 날짜 공통 영역만 (좌표가 다르면 밀린 만큼 맞춤). 크기가 1025 인 경우도 처리
            h = min(i1.shape[0], l1.shape[0], i2.shape[0] + dy, l2.shape[0] + dy) - max(dy, 0)
            w = min(i1.shape[1], l1.shape[1], i2.shape[1] + dx, l2.shape[1] + dx) - max(dx, 0)
            y0, x0 = max(dy, 0), max(dx, 0)
            A = np.s_[y0:y0 + h, x0:x0 + w]
            B = np.s_[y0 - dy:y0 - dy + h, x0 - dx:x0 - dx + w]
            i1, l1, i2, l2 = i1[A], l1[A], i2[B], l2[B]
            bys, bxs = np.nonzero(l2 == BUILDING)
            for k in range(a.windows):
                win = int(round(256 * rng.uniform(0.55, 0.7) / 0.5))
                if h < win or w < win:
                    break
                if len(bys) and rng.random() < a.building_focus:      # 건물 하나를 창 안 무작위 위치에
                    j = rng.integers(len(bys))
                    y = int(np.clip(bys[j] - rng.integers(win // 4, 3 * win // 4), 0, h - win))
                    x = int(np.clip(bxs[j] - rng.integers(win // 4, 3 * win // 4), 0, w - win))
                else:
                    y, x = rng.integers(0, h - win + 1), rng.integers(0, w - win + 1)
                s = np.s_[y:y + win, x:x + win]
                pre, post, la, lb = i1[s].copy(), i2[s].copy(), l1[s], l2[s]
                ign = (la == NONTARGET) | (lb == NONTARGET)
                if ign.mean() > a.max_ignore:
                    stat["skip_nontarget"] += 1
                    continue
                pre, new_b = synth_new_building(pre, lb, rng, a.p_build)
                forest_both = ((la == FOREST) & (lb == FOREST)).astype(np.uint8)
                post, cut = synth_tree_removal(post, forest_both, rng, a.p_tree,
                                               lo=int(0.03 * win * win), hi=int(0.15 * win * win))
                pre, _ = to_patch(pre, [])
                post, (mb, mt, mi) = to_patch(post, [new_b, cut, ign])
                mi = mi.astype(bool) & ~(mb.astype(bool) | mt.astype(bool))
                save(a.out, f"{t}_{d1}_{d2}_{k}", pre, post, mb, mt, mi)
                stat["pairs"] += 1
                stat["with_building"] += int(mb.any())
                stat["with_tree"] += int(mt.any())
                stat["no_change"] += int(not (mb.any() or mt.any()))
        stat["tiles"] += 1
        if stat["tiles"] % 50 == 0:
            print(dict(stat), flush=True)
    print("완료", dict(stat))


if __name__ == "__main__":
    main()
