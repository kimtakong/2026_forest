"""전체 파이프라인 재현: python run_all.py [--sample] [--from 01] [--only 00]

--sample : 오래 걸리는 단계(06 등)를 표본으로 빠르게 시연
--from   : 해당 번호부터 실행
--only   : 해당 번호만 실행
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    ("dl", "download_boundary.py"),
    ("00", "00_data_checks.py"),
    ("01", "01_prepare.py"),
    ("02", "02_descriptive.py"),
    ("03", "03_official_vs_actual.py"),
    ("04", "04_cif_vs_naive.py"),
    ("05", "05_conditional_residual.py"),
    ("06", "06_hazard_model.py"),
    ("07", "07_equity_map.py"),
    ("08", "08_policy_simulation.py"),
    ("09", "09_deadhead_carbon.py"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    a = ap.parse_args()
    keys = [k for k, _ in STEPS]
    steps = STEPS
    if a.only:
        steps = [s for s in STEPS if s[0] == a.only]
    elif a.start:
        steps = STEPS[keys.index(a.start):]
    t_all = time.time()
    timing = []
    for key, script in steps:
        path = SRC / script
        if not path.exists():
            print(f"[skip] {script} (아직 없음)")
            continue
        print(f"\n===== {key}: {script} =====", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(path)] + (["--sample"] if a.sample else [])
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        subprocess.run(cmd, cwd=SRC, check=True, env=env)
        timing.append((script, time.time() - t0))
    print("\n===== 실행 시간 =====")
    for s, t in timing:
        print(f"  {s:<32s} {t:7.1f}초")
    print(f"  {'합계':<32s} {time.time() - t_all:7.1f}초")


if __name__ == "__main__":
    main()
