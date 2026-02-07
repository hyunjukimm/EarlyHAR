#!/usr/bin/env python3
"""
결과 파일 포맷 일치 여부 확인.
모든 baseline의 kfold_summary.csv가 collect_f1_earliness_hm.py에서 읽을 수 있는 구조인지 검증.
"""
import pandas as pd
from pathlib import Path

RESULTS = Path("results")
DATASETS = ["aras", "casas", "doore", "openpack", "opportunity"]
BASELINES = ["calimera", "dc", "earliest", "stopandhop", "attn", "teaser", "lecgan"]
REQUIRED_METRICS = ["accuracy", "f1", "earliness", "f_e"]
REQUIRED_COLS = ["mean", "std"]

def check_one(path: Path):
    """단일 CSV 검사. (ok, msg) 반환."""
    if not path.exists():
        return False, "파일 없음"
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as e:
        return False, f"읽기 실패: {e}"
    # index가 metric 이름
    idx = df.index.tolist()
    has_metrics = all(m in idx for m in REQUIRED_METRICS)
    has_cols = all(c in df.columns for c in REQUIRED_COLS)
    if not has_metrics:
        miss = [m for m in REQUIRED_METRICS if m not in idx]
        return False, f"필수 metric 없음: {miss}"
    if not has_cols:
        miss = [c for c in REQUIRED_COLS if c not in df.columns]
        return False, f"필수 column 없음: {miss}"
    return True, "OK"

def main():
    print("=== 결과 포맷 검증 ===\n")
    all_ok = True
    for ds in DATASETS:
        for bl in BASELINES:
            p = RESULTS / ds / f"{bl}_kfold_summary.csv"
            ok, msg = check_one(p)
            status = "✓" if ok else "✗"
            print(f"  {status} {ds:12} {bl:12} {msg}")
            if not ok:
                all_ok = False

    print("\n=== 포맷 요약 ===")
    print("공통 요구사항:")
    print("  - index: metric 이름 (accuracy, f1, earliness, f_e 등)")
    print("  - columns: mean, std")
    print("  - collect_f1_earliness_hm.py 가 index_col=0 으로 읽음")
    print(f"\n일치 여부: {'모두 동일 포맷' if all_ok else '일부 불일치'}")
    return 0 if all_ok else 1

if __name__ == "__main__":
    exit(main())
