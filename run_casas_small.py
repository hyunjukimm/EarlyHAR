#!/usr/bin/env python3
"""
casas 데이터셋으로 모든 baseline 작은 테스트 실행.
--sample_ratio 0.05, --max_train 30, --max_test 20

사용법: conda activate <env> 후  python run_casas_small.py
"""
import subprocess
import sys
from pathlib import Path

BASELINES = [
    ("calimera", "python main_cal.py"),
    ("dc", "python main_dc.py"),
    ("earliest", "python main_earliest.py"),
    ("stopandhop", "python main_stopandhop.py"),
    ("attn", "python main_attn.py"),
    ("teaser", "python main_teaser.py"),
    ("lecgan", "python main_lecgan.py"),
]
ARGS = "--dataset casas --sample_ratio 0.05 --max_train 30 --max_test 20"

def main():
    root = Path(__file__).resolve().parent
    for name, cmd in BASELINES:
        full = f"{cmd} {ARGS}"
        print(f"\n{'='*50}")
        print(f"[casas] {name}")
        print(f"{'='*50}")
        ret = subprocess.run(full, shell=True, cwd=root, timeout=300)
        if ret.returncode != 0:
            print(f"\nFAILED: {name}")
            return 1
    print(f"\n\n모든 baseline 성공 (casas)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
