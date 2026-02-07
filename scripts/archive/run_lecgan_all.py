#!/usr/bin/env python3
"""
LEC-GAN 전체 데이터셋 순차 실행.
GPU: 사용 가능 시 여유 메모리가 가장 많은 GPU를 자동 선택, 아니면 --gpu로 지정.
p_thresh: 기본 후보(0.75,0.8,0.85,0.9,0.95)로 val F-E 기준 sweep.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 프로젝트 루트
ROOT = Path(__file__).resolve().parent
DATASETS = ["aras", "casas", "doore", "opportunity", "openpack"]
P_THRESH_SWEEP_DEFAULT = "0.75,0.8,0.85,0.9,0.95"


def pick_best_gpu():
    """nvidia-smi로 여유 메모리가 가장 많은 GPU 인덱스 반환. 실패 시 0."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return 0
        best_idx, best_free = 0, -1
        for line in out.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 2:
                idx = int(parts[0].strip())
                free = int(parts[1].strip().split()[0] or "0")
                if free > best_free:
                    best_free = free
                    best_idx = idx
        return best_idx
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description="LEC-GAN 전체 데이터셋 실행 (GPU 자동 선택)")
    parser.add_argument("--gpu", type=int, default=None, help="고정 GPU ID (미지정 시 여유 메모리 최대 GPU)")
    parser.add_argument("--cpu", action="store_true", help="CPU만 사용")
    parser.add_argument("--p_thresh_sweep", type=str, default=P_THRESH_SWEEP_DEFAULT, help=f"p_thresh 후보 (기본: {P_THRESH_SWEEP_DEFAULT})")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--datasets", type=str, nargs="*", default=DATASETS, help=f"실행할 데이터셋 (기본: {' '.join(DATASETS)})")
    parser.add_argument("--nice", type=int, default=19, help="우선순위 낮춤 (0=안 함, 1–19). 기본 19로 다른 실험에 영향 최소화")
    parser.add_argument("--num_threads", type=int, default=4, help="LEC-GAN용 OMP/MKL 스레드 수 (다른 실험과 CPU 덜 겹치게)")
    args = parser.parse_args()

    os.chdir(ROOT)
    os.makedirs("logs", exist_ok=True)

    if args.cpu:
        gpu_id = None
        device_args = ["--device", "cpu"]
        print("[LEC-GAN] CPU 모드로 실행")
    else:
        gpu_id = args.gpu if args.gpu is not None else pick_best_gpu()
        device_args = ["--device", "cuda"]
        print(f"[LEC-GAN] GPU {gpu_id} 사용 (CUDA_VISIBLE_DEVICES={gpu_id})")
    print(f"[LEC-GAN] p_thresh_sweep: {args.p_thresh_sweep}")
    print(f"[LEC-GAN] 데이터셋: {args.datasets}")
    if args.nice > 0:
        print(f"[LEC-GAN] nice={args.nice} (다른 실험에 영향 최소화), OMP/MKL 스레드={args.num_threads}")
    print("=" * 50)

    for dataset in args.datasets:
        log_file = ROOT / "logs" / f"lecgan_{dataset}.log"
        cmd = [
            sys.executable,
            str(ROOT / "main_lecgan.py"),
            "--dataset", dataset,
            "--k_fold", str(args.k_fold),
            "--epochs", str(args.epochs),
            "--p_thresh_sweep", args.p_thresh_sweep,
            "--padding", "mean",
        ] + device_args
        print(f"\n▶ [{dataset}] 시작 (로그: {log_file})")
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["OMP_NUM_THREADS"] = str(args.num_threads)
        env["MKL_NUM_THREADS"] = str(args.num_threads)
        env["NUMEXPR_NUM_THREADS"] = str(args.num_threads)
        run_cmd = cmd
        if args.nice > 0:
            run_cmd = ["nice", "-n", str(args.nice)] + cmd
        with open(log_file, "w") as f:
            ret = subprocess.run(run_cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        if ret.returncode == 0:
            print(f"✅ 완료: {dataset}")
        else:
            print(f"❌ 실패: {dataset} (exit {ret.returncode}, 로그: {log_file})")

    print("\n" + "=" * 50)
    print("LEC-GAN 전체 실행 종료. 결과: results/*/lecgan_*.csv")


if __name__ == "__main__":
    main()
