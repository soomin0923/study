"""외부 데이터 -> 학습용 전/후 쌍 (256x256, 1화소 0.55~0.7m).

두 가지 방식이 있습니다.

[pair] 실제 전/후 영상과 변화 라벨이 있는 경우 (AIHub 변화탐지 위성 데이터 등)
  python scripts/prepare_aihub.py pair --pre <전영상폴더> --post <후영상폴더> --labels <변화라벨PNG폴더> \
      --building-values 1 --tree-values 2 --src-gsd 0.5 --out data/aihub_cd
  세 폴더의 같은 파일명(확장자 무관)을 한 쌍으로 봅니다. 라벨은 후 영상 격자의 클래스 마스크 PNG.

[synth] 한 시점 영상 + 건물/산림 마스크만 있는 경우 -> 변화를 합성
  python scripts/prepare_aihub.py synth --images <영상> --labels <클래스마스크> \
      --building-values 3 --forest-values 5 --src-gsd 0.5 --out data/synth_cd
  - 증축: 건물 일부를 '전' 영상에서 지워(inpaint) 전에는 없던 건물로 만듭니다
  - 벌목: 산림 안의 덩어리를 '후' 영상에서 맨땅/풀밭 질감으로 바꿉니다
  합성 변화는 실제와 모양이 다르므로 보조 자료입니다. 효과는 검증셋 점수로 판단하십시오.

라벨이 폴리곤 JSON 이면 먼저 클래스 마스크 PNG 로 래스터화하십시오 (AIHub 라벨 형식은 받아서 확인 필요).
AIHub 이용약관이 대회 참가·파생물 제출을 허용하는지 반드시 확인하십시오.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT  # noqa: F401

EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def index(folder):
    return {p.stem: p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS}


def imread_rgb(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def imread_lab(p):
    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    return None if m is None else (m[..., 0] if m.ndim == 3 else m)


def windows(h, w, win, stride):
    for y in range(0, max(1, h - win + 1), stride):
        for x in range(0, max(1, w - win + 1), stride):
            yield y, x


def resize_to_patch(img, mask_list):
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    ms = [(cv2.resize(m.astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) >= 0.5).astype(np.uint8)
          for m in mask_list]
    return img, ms


def save_pair(out, name, pre, post, bld, tree):
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "pre.png"), pre[..., ::-1])
    cv2.imwrite(str(d / "post.png"), post[..., ::-1])
    cv2.imwrite(str(d / "building.png"), bld * 255)
    cv2.imwrite(str(d / "tree.png"), tree * 255)


def bare_texture(rng, shape):
    base = rng.choice([[176, 142, 96], [150, 120, 84], [168, 176, 120], [190, 170, 130]]).astype(np.float32)
    base = base * rng.uniform(0.85, 1.15)
    noise = cv2.GaussianBlur(rng.normal(0, 14, shape[:2]).astype(np.float32), (0, 0), 2.0)
    return np.clip(base[None, None] + noise[..., None], 0, 255)


def random_blob_in(mask, rng, lo=300, hi=3000):
    """mask 안에 들어가는 불규칙 덩어리 (벌목지 흉내, 원본 해상도 기준)."""
    ys, xs = np.nonzero(mask)
    if len(ys) < lo:
        return None
    k = rng.integers(len(ys))
    cy, cx = ys[k], xs[k]
    r = np.sqrt(rng.uniform(lo, hi) / np.pi)
    blob = np.zeros(mask.shape, np.uint8)
    pts = []
    for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        rr = r * rng.uniform(0.6, 1.4)
        pts.append([cx + rr * np.cos(t), cy + rr * np.sin(t)])
    cv2.fillPoly(blob, [np.array(pts, np.int32)], 1)
    blob &= mask.astype(np.uint8)
    return blob if blob.sum() >= lo * 0.5 else None


def run_pair(a, rng):
    pres, posts, labs = index(a.pre), index(a.post), index(a.labels)
    n = 0
    for stem, lp in sorted(labs.items()):
        if stem not in pres or stem not in posts:
            continue
        pre, post, lab = imread_rgb(pres[stem]), imread_rgb(posts[stem]), imread_lab(lp)
        if pre is None or post is None or lab is None or pre.shape != post.shape:
            continue
        for y, x in windows(*lab.shape, int(round(256 * rng.uniform(0.55, 0.7) / a.src_gsd)), a.stride):
            win = int(round(256 * rng.uniform(0.55, 0.7) / a.src_gsd))
            sl = np.s_[y:y + win, x:x + win]
            if lab[sl].shape != (win, win):
                continue
            b, t = np.isin(lab[sl], a.building_values), np.isin(lab[sl], a.tree_values)
            if not (b.any() or t.any()) and rng.random() > a.neg_keep:
                continue
            p1, _ = resize_to_patch(pre[sl], [])
            p2, (mb, mt) = resize_to_patch(post[sl], [b, t])
            save_pair(a.out, f"{stem}_{y}_{x}", p1, p2, mb, mt)
            n += 1
    print("pairs", n)


def run_synth(a, rng):
    imgs, labs = index(a.images), index(a.labels)
    n = 0
    for stem, lp in sorted(labs.items()):
        if stem not in imgs:
            continue
        img, lab = imread_rgb(imgs[stem]), imread_lab(lp)
        if img is None or lab is None or img.shape[:2] != lab.shape:
            continue
        for y, x in windows(*lab.shape, int(round(256 * 0.62 / a.src_gsd)), a.stride):
            win = int(round(256 * rng.uniform(0.55, 0.7) / a.src_gsd))
            sl = np.s_[y:y + win, x:x + win]
            if lab[sl].shape != (win, win):
                continue
            im, lb = img[sl].copy(), lab[sl]
            bmask = np.isin(lb, a.building_values).astype(np.uint8)
            fmask = np.isin(lb, a.forest_values).astype(np.uint8)
            pre, post = im.copy(), im.copy()
            new_b = np.zeros_like(bmask)
            nb, blab = cv2.connectedComponents(bmask)
            for k in range(1, nb):
                if rng.random() < a.p_build:
                    new_b |= (blab == k).astype(np.uint8)
            if new_b.any():   # 전 영상에서 건물(+그림자 여유) 지우기
                hole = cv2.dilate(new_b, np.ones((5, 5), np.uint8))
                pre = cv2.inpaint(pre, hole, 7, cv2.INPAINT_TELEA)
            cut = random_blob_in(fmask & (1 - bmask), rng) if rng.random() < a.p_tree else None
            if cut is not None:   # 후 영상에서 산림 덩어리를 맨땅으로
                tex = bare_texture(rng, post.shape)
                soft = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 1.0)[..., None]
                post = (post * (1 - soft) + tex * soft).astype(np.uint8)
            else:
                cut = np.zeros_like(fmask)
            if not new_b.any() and not cut.any() and rng.random() > a.neg_keep:
                continue
            p1, _ = resize_to_patch(pre, [])
            p2, (mb, mt) = resize_to_patch(post, [new_b, cut])
            save_pair(a.out, f"{stem}_{y}_{x}", p1, p2, mb, mt)
            n += 1
    print("synthetic pairs", n)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("pair", "synth"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True)
        s.add_argument("--labels", required=True)
        s.add_argument("--building-values", type=int, nargs="+", required=True)
        s.add_argument("--src-gsd", type=float, default=0.5)
        s.add_argument("--stride", type=int, default=256)
        s.add_argument("--neg-keep", type=float, default=0.2, help="변화 없는 창을 남길 확률")
        s.add_argument("--seed", type=int, default=0)
    sp, ss = sub.choices["pair"], sub.choices["synth"]
    sp.add_argument("--pre", required=True)
    sp.add_argument("--post", required=True)
    sp.add_argument("--tree-values", type=int, nargs="+", required=True)
    ss.add_argument("--images", required=True)
    ss.add_argument("--forest-values", type=int, nargs="+", required=True)
    ss.add_argument("--p-build", type=float, default=0.3, help="건물 하나를 '새 건물'로 만들 확률")
    ss.add_argument("--p-tree", type=float, default=0.4, help="창 하나에 벌목을 합성할 확률")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    (run_pair if a.mode == "pair" else run_synth)(a, rng)


if __name__ == "__main__":
    main()
