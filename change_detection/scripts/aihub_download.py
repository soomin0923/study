"""AIHub API 다운로드를 파이썬만으로 (윈도우·맥·리눅스, WSL 불필요).

AIHub 공식 도구(aihubshell)는 bash 스크립트라 윈도우에서 바로 못 씁니다. 이 모듈은
1) aihubshell 스크립트를 내려받아 그 안의 다운로드 주소를 읽고 (못 찾으면 알려진 기본 주소)
2) API 키를 헤더로 붙여 파일키별로 내려받은 뒤 (진행률 표시)
3) tar 로 오면 풀고, 분할 조각(*.zip.partN)을 합치고, zip 을 풉니다.

주의: AIHub 는 해외 IP 다운로드를 막습니다. 국내 PC에서 실행하십시오.
      다운로드 승인 + 마이페이지 API 키가 필요합니다.

  python scripts/aihub_download.py --datasetkey 71363 --filekeys 491174 491178 --out aihub_71363
  (키는 실행 중에 입력받습니다. 환경변수 AIHUB_API_KEY 가 있으면 그것을 씁니다)
"""
import argparse
import getpass
import os
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

SHELL_URL = "https://api.aihub.or.kr/api/aihubshell.do"
DEFAULT_DOWN = "https://api.aihub.or.kr/down/0.6/{datasetkey}.do?fileSn={filekey}"


def fetch_shell(out: Path) -> str:
    """공식 aihubshell 스크립트를 받아 저장하고 내용을 돌려줍니다 (다운로드 주소 확인용)."""
    r = requests.get(SHELL_URL, timeout=30)
    r.raise_for_status()
    out.mkdir(parents=True, exist_ok=True)
    (out / "aihubshell").write_bytes(r.content)
    return r.text


def down_template(shell_text: str | None) -> tuple[str, str]:
    """스크립트에서 다운로드 주소 틀을 찾습니다. (틀, 출처 설명)"""
    if shell_text:
        m = re.search(r"https?://api\.aihub\.or\.kr/down/[0-9.]+/", shell_text)
        if m:
            return m.group(0) + "{datasetkey}.do?fileSn={filekey}", "aihubshell 스크립트에서 찾음"
    return DEFAULT_DOWN, "기본 주소 (스크립트에서 못 찾음 - 실패하면 show_shell_lines() 결과를 확인)"


def shell_lines(shell_text: str, pat=r"curl|down|apikey|BASE"):
    """스크립트에서 주소·헤더 관련 줄만 보여줍니다 (키 값은 스크립트에 없음)."""
    return [ln.rstrip() for ln in shell_text.splitlines() if re.search(pat, ln, re.I)][:40]


def _fmt(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.1f}{u}" if u != "B" else f"{n}B"
        n /= 1024


def download_one(template, datasetkey, filekey, apikey, out: Path, chunk=8 << 20) -> Path:
    """파일키 하나를 받습니다. 실패하면 서버가 준 메시지와 함께 RuntimeError."""
    url = template.format(datasetkey=datasetkey, filekey=filekey)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"download_{datasetkey}_{filekey}.tar"
    with requests.get(url, headers={"apikey": apikey}, stream=True, timeout=(15, 120)) as r:
        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "json" in ctype or "text/html" in ctype:
            body = r.content[:2000].decode("utf-8", "replace")
            raise RuntimeError(f"[{filekey}] HTTP {r.status_code}: {body.strip()}")
        total = int(r.headers.get("Content-Length", 0))
        done, t0, last = 0, time.time(), 0.0
        with open(dst, "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
                done += len(part)
                now = time.time()
                if now - last > 5 or (total and done == total):
                    speed = done / max(now - t0, 1e-6)
                    pct = f"{100 * done / total:5.1f}%" if total else ""
                    eta = f" 남은 {int((total - done) / speed // 60)}분" if total and speed > 0 else ""
                    print(f"  [{filekey}] {pct} {_fmt(done)}{'/' + _fmt(total) if total else ''} "
                          f"{_fmt(speed)}/s{eta}", flush=True)
                    last = now
    if dst.stat().st_size == 0:
        raise RuntimeError(f"[{filekey}] 0바이트를 받았습니다")
    return dst


def unpack(path: Path, out: Path, delete=True):
    """받은 파일이 tar 이면 풀고 지웁니다 (AIHub API 는 tar 로 묶어 보내는 경우가 많음)."""
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            t.extractall(out, filter="data") if hasattr(tarfile, "data_filter") else t.extractall(out)
        if delete:
            path.unlink()
        return "tar"
    if zipfile.is_zipfile(path):
        new = path.with_suffix(".zip")
        path.rename(new)
        return "zip"
    return "unknown"


def merge_parts(root: Path, delete=True):
    """*.zip.partN 을 번호 순으로 합칩니다 (scripts/merge_parts.py 와 같은 동작)."""
    groups = {}
    for p in root.rglob("*.part*"):
        m = re.match(r"(.*)\.part(\d+)$", str(p))
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), p))
    merged = []
    for target, parts in sorted(groups.items()):
        parts.sort()
        with open(target, "wb") as out:
            for _, p in parts:
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out, 16 << 20)
        size = Path(target).stat().st_size
        print(f"  병합 {Path(target).name}: 조각 {len(parts)}개 -> {_fmt(size)}")
        if size == 0:
            raise RuntimeError("병합 결과 0바이트")
        if delete:
            for _, p in parts:
                p.unlink()
        merged.append(Path(target))
    return merged


def extract_zips(root: Path, delete=True):
    for z in sorted(root.rglob("*.zip")):
        dst = z.with_suffix("")
        if dst.exists():
            continue
        print("  압축 해제", z.name, flush=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst)
        if delete:
            z.unlink()


def run(datasetkey, filekeys, apikey, out, template=None, delete=True):
    out = Path(out)
    if template is None:
        try:
            template, how = down_template(fetch_shell(out))
        except requests.RequestException as e:
            template, how = DEFAULT_DOWN, f"기본 주소 (aihubshell 받기 실패: {e})"
        print("다운로드 주소:", template.split("{")[0], "-", how)
    free = shutil.disk_usage(out if out.exists() else out.parent).free
    print(f"디스크 여유 {_fmt(free)}")
    for fk in filekeys:
        print(f"== 파일키 {fk} 받는 중", flush=True)
        p = download_one(template, datasetkey, fk, apikey, out)
        kind = unpack(p, out, delete)
        print(f"  받은 형식: {kind}")
    merge_parts(out, delete)
    extract_zips(out, delete)
    print("완료:", out.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasetkey", default="71363")
    ap.add_argument("--filekeys", nargs="+", required=True)
    ap.add_argument("--out", default="aihub_71363")
    ap.add_argument("--keep", action="store_true", help="조각·zip 을 지우지 않음")
    a = ap.parse_args()
    key = os.environ.get("AIHUB_API_KEY") or getpass.getpass("AIHub API 키: ")
    run(a.datasetkey, a.filekeys, key.strip().strip("'\""), a.out, delete=not a.keep)


if __name__ == "__main__":
    main()
