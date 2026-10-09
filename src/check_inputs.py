"""입력 원본 확인: datas/ 의 원본 CSV 4개가 이 분석에 쓴 파일과 같은지 본다.

docs/input_manifest.csv 에 적힌 크기·줄 수·SHA-256과 비교한다. 다르면 경고만 하고 계속한다.
공공데이터포털 파일이 갱신되면 결과가 보고서 숫자와 달라질 수 있기 때문이다.
  python src/check_inputs.py          # 확인
  python src/check_inputs.py --write  # 지금 datas/ 의 파일로 기준값을 기록(분석에 쓴 원본이 있는 PC에서 한 번)
"""
import argparse
import hashlib
import sys
import time

import pandas as pd

from utils import INPUT_MANIFEST, RAW_DISABILITY, RAW_OFFICIAL_WAIT, RAW_PURPOSE, RAW_TRIPS, ROOT

RAW_FILES = [RAW_TRIPS, RAW_OFFICIAL_WAIT, RAW_PURPOSE, RAW_DISABILITY]
COLS = ["파일", "크기(바이트)", "줄 수(머리행 포함)", "SHA-256"]


def describe(path):
    """파일 크기, 줄 수(마지막 줄에 줄바꿈이 없어도 한 줄로 셈), SHA-256."""
    h = hashlib.sha256()
    lines, last = 0, b"\n"
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    if last != b"\n":
        lines += 1
    return {"파일": path.name, "크기(바이트)": path.stat().st_size, "줄 수(머리행 포함)": lines, "SHA-256": h.hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="지금 datas/ 의 파일로 기준값을 기록한다")
    ap.add_argument("--sample", action="store_true", help="run_all.py 호환용(무시)")
    a = ap.parse_args()
    t0 = time.time()

    missing = [p.name for p in RAW_FILES if not p.exists()]
    if missing:
        print("[check_inputs] datas/ 에 원본 파일이 없다. README '데이터' 절의 URL에서 받아 파일명 그대로 넣는다.")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    now = pd.DataFrame([describe(p) for p in RAW_FILES], columns=COLS)
    if a.write:
        now.to_csv(INPUT_MANIFEST, index=False, encoding="utf-8-sig")
        print(now.to_string(index=False))
        print(f"  -> {INPUT_MANIFEST.relative_to(ROOT)} 기록. 이 파일을 커밋한다.")
    elif not INPUT_MANIFEST.exists():
        print(f"  기준값 파일({INPUT_MANIFEST.relative_to(ROOT)})이 없어 비교하지 않는다. "
              "분석에 쓴 원본이 있는 PC에서 'python src/check_inputs.py --write'로 만든다.")
    else:
        ref = pd.read_csv(INPUT_MANIFEST, encoding="utf-8-sig", dtype={"SHA-256": str}).set_index("파일")
        differ = []
        for r in now.itertuples(index=False):
            name, size, lines, sha = r
            if name not in ref.index:
                print(f"  [기준값 없음] {name}")
                continue
            e = ref.loc[name]
            if sha == e["SHA-256"]:
                print(f"  [같음] {name}")
            else:
                differ.append(name)
                print(f"  [다름] {name}: 크기 {e['크기(바이트)']:,} -> {size:,}, "
                      f"줄 수 {e['줄 수(머리행 포함)']:,} -> {lines:,}")
        if differ:
            print("  [경고] 위 파일이 이 분석에 쓴 원본과 다르다. 공공데이터포털에서 파일이 갱신됐을 수 있다.\n"
                  "         실행은 계속하지만 표·그림 숫자가 보고서와 다를 수 있다.")
    print(f"[check_inputs] 완료 {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
