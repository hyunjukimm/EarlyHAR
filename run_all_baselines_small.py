#!/usr/bin/env python3
"""
모든 baseline을 모든 데이터셋에 아주 작은 테스트로 실행.
--sample_ratio 0.05, --max_train 30, --max_test 20 사용.

사용법: 가상환경 활성화 후 실행
  conda activate <env>   # 또는 source venv/bin/activate
  python run_all_baselines_small.py

결과 포맷 검증: python verify_result_format.py
"""
import subprocess
import sys
from pathlib import Path

DATASETS = ["aras", "casas", "doore", "openpack", "opportunity"]
BASELINES = [
    ("calimera", "python main_cal.py"),
    ("dc", "python main_dc.py"),
    ("earliest", "python main_earliest.py"),
    ("stopandhop", "python main_stopandhop.py"),
    ("attn", "python main_attn.py"),
    ("teaser", "python main_teaser.py"),
    ("lecgan", "python main_lecgan.py"),
]
QUICK_ARGS = "--sample_ratio 0.05 --max_train 30 --max_test 20"

def main():
    root = Path(__file__).resolve().parent
    ok = 0
    fail = 0
    for ds in DATASETS:
        for name, cmd in BASELINES:
            full_cmd = f"{cmd} --dataset {ds} {QUICK_ARGS}"
            print(f"\n{'='*60}")
            print(f"[{ds}] {name}")
            print(f"{'='*60}")
            ret = subprocess.run(
                full_cmd,
                shell=True,
                cwd=root,
                timeout=300,
            )
            if ret.returncode == 0:
                ok += 1
            else:
                fail += 1
                print(f"FAILED: {name} on {ds}")

    print(f"\n\nDone: {ok} ok, {fail} failed")
    if ok > 0:
        print("\n결과 포맷 검증: python verify_result_format.py")
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
