"""AIHub 분할 파일(*.zip.part0, part1, ...) 병합 + 압축 해제. 윈도우·맥·리눅스 공통 (WSL 불필요).

AIHub 안내의 리눅스 명령 `find ... | sort -k2V | xargs cat > 파일.zip` 과 같은 동작입니다.

  python scripts/merge_parts.py <다운로드폴더>              # 병합 + 해제, 조각은 남김
  python scripts/merge_parts.py <다운로드폴더> --delete     # 병합·해제 후 조각과 zip 삭제 (디스크 절약)
"""
import argparse
import re
import zipfile
from pathlib import Path


def merge(root: Path, delete: bool) -> list[Path]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for p in root.rglob("*.part*"):
        m = re.match(r"(.*)\.part(\d+)$", str(p))
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
    merged = []
    for target, parts in sorted(groups.items()):
        parts.sort()                                    # part2 < part10 (숫자 순)
        nums = [n for n, _ in parts]
        if nums != list(range(nums[0], nums[0] + len(nums))):
            print(f"경고: {Path(target).name} 조각 번호가 비어 있습니다 {nums} - 다운로드가 덜 됐을 수 있습니다")
        with open(target, "wb") as out:
            for _, p in parts:
                with open(p, "rb") as f:
                    while chunk := f.read(1 << 24):
                        out.write(chunk)
        size = Path(target).stat().st_size
        print(f"병합 {Path(target).name}: 조각 {len(parts)}개 -> {size / 2**20:.1f} MB")
        if size == 0:
            raise SystemExit("병합 결과가 0바이트입니다. 폴더 경로를 확인하세요")
        if delete:
            for _, p in parts:
                p.unlink()
        merged.append(Path(target))
    return merged


def extract(root: Path, delete: bool):
    for z in sorted(root.rglob("*.zip")):
        dst = z.with_suffix("")
        if dst.exists():
            continue
        print("압축 해제", z.name)
        with zipfile.ZipFile(z) as zf:
            bad = zf.testzip()
            if bad:
                raise SystemExit(f"{z.name}: 손상된 항목 {bad} - 다시 받으십시오")
            zf.extractall(dst)
        if delete:
            z.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--delete", action="store_true", help="처리 후 조각과 zip 삭제")
    ap.add_argument("--no-extract", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    merge(root, a.delete)
    if not a.no_extract:
        extract(root, a.delete)


if __name__ == "__main__":
    main()
