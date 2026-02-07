#!/usr/bin/env python3
"""
동적 GPU 스케줄러 - 가장 효율적인 병렬 실행
GPU 4개에 15개 작업을 동적으로 할당하여 최대 효율 달성
"""

import subprocess
import os
import time
from datetime import datetime
from pathlib import Path
import queue
import threading

# 실험 조합 정의
DATASETS = ["aras", "casas", "doore", "openpack", "opportunity"]
BASELINES = ["calimera", "earliest", "stopandhop"]

# 예상 실행 시간 (분 단위, 실제 측정값 기반)
ESTIMATED_TIME = {
    ("calimera", "aras"): 10,
    ("calimera", "casas"): 15,
    ("calimera", "doore"): 5,
    ("calimera", "openpack"): 20,
    ("calimera", "opportunity"): 12,
    ("earliest", "aras"): 5,
    ("earliest", "casas"): 7,
    ("earliest", "doore"): 2,
    ("earliest", "openpack"): 10,
    ("earliest", "opportunity"): 6,
    ("stopandhop", "aras"): 15,
    ("stopandhop", "casas"): 20,
    ("stopandhop", "doore"): 8,
    ("stopandhop", "openpack"): 30,
    ("stopandhop", "opportunity"): 18,
}

# 공통 파라미터
PARAMS = {
    "k_fold": 5,
    "padding": "mean",
    "augment": "True",
    "n_epochs": 50,
    "batch_size": 32,
    "nhid": 64,
    "patience": 10,
    "delay_penalty": 1,
}

class GPUWorker(threading.Thread):
    """각 GPU에서 작업을 순차적으로 실행하는 워커"""
    
    def __init__(self, gpu_id, task_queue, log_dir):
        super().__init__()
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.log_dir = log_dir
        self.completed_tasks = []
        
    def run(self):
        """작업 큐에서 작업을 가져와 실행"""
        print(f"[GPU {self.gpu_id}] 워커 시작")
        
        while True:
            try:
                # 작업 가져오기 (타임아웃 1초)
                baseline, dataset = self.task_queue.get(timeout=1)
                
                # 작업 실행
                start_time = time.time()
                print(f"\n[GPU {self.gpu_id} - {datetime.now().strftime('%H:%M:%S')}] "
                      f"{baseline.upper()} - {dataset} 시작")
                
                self._run_experiment(baseline, dataset)
                
                elapsed = (time.time() - start_time) / 60
                print(f"[GPU {self.gpu_id} - {datetime.now().strftime('%H:%M:%S')}] "
                      f"{baseline.upper()} - {dataset} 완료 (소요: {elapsed:.1f}분)")
                
                self.completed_tasks.append((baseline, dataset, elapsed))
                self.task_queue.task_done()
                
            except queue.Empty:
                # 큐가 비어있으면 종료
                break
                
        print(f"\n[GPU {self.gpu_id}] 워커 종료 - 완료 작업: {len(self.completed_tasks)}개")
    
    def _run_experiment(self, baseline, dataset):
        """개별 실험 실행"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        env["NUMBA_CACHE_DIR"] = "/home/juice/EarlyHAR/EarlyHAR/.numba_cache"
        
        # conda 환경 경로 설정 (pytorch-env)
        conda_python = "/home/juice/miniconda3/envs/pytorch-env/bin/python"
        
        log_file = self.log_dir / f"gpu{self.gpu_id}_{baseline}_{dataset}.log"
        
        # 명령어 구성 (conda 환경의 python 사용)
        if baseline == "calimera":
            cmd = [
                conda_python, "main_cal.py",
                "--dataset", dataset,
                "--k_fold", str(PARAMS["k_fold"]),
                "--padding", PARAMS["padding"],
                "--augment", PARAMS["augment"],
                "--delay_penalty", str(PARAMS["delay_penalty"]),
            ]
        elif baseline == "earliest":
            cmd = [
                conda_python, "main_earliest.py",
                "--dataset", dataset,
                "--k_fold", str(PARAMS["k_fold"]),
                "--n_epochs", str(PARAMS["n_epochs"]),
                "--batch_size", str(PARAMS["batch_size"]),
                "--nhid", str(PARAMS["nhid"]),
                "--patience", str(PARAMS["patience"]),
                "--padding", PARAMS["padding"],
                "--augment", PARAMS["augment"],
            ]
        else:  # stopandhop
            cmd = [
                conda_python, "main_stopandhop.py",
                "--dataset", dataset,
                "--k_fold", str(PARAMS["k_fold"]),
                "--n_epochs", str(PARAMS["n_epochs"]),
                "--batch_size", str(PARAMS["batch_size"]),
                "--nhid", str(PARAMS["nhid"]),
                "--patience", str(PARAMS["patience"]),
                "--padding", PARAMS["padding"],
                "--augment", PARAMS["augment"],
            ]
        
        # 실행
        with open(log_file, "w") as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd="/home/juice/EarlyHAR/EarlyHAR")

def main():
    """메인 실행 함수"""
    print("="*80)
    print("동적 GPU 스케줄러 시작")
    print("="*80)
    print(f"GPU 개수: 4개")
    print(f"전체 작업: 15개 (5 datasets × 3 baselines)")
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 로그 디렉토리 생성
    log_dir = Path("logs") / f"dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n로그 저장: {log_dir}")
    
    # 작업 큐 생성 (예상 시간 역순으로 정렬 - 긴 작업 먼저)
    task_queue = queue.Queue()
    
    tasks = [(baseline, dataset) for baseline in BASELINES for dataset in DATASETS]
    tasks.sort(key=lambda x: ESTIMATED_TIME.get(x, 10), reverse=True)
    
    print("\n작업 순서 (예상 시간 기준):")
    for i, (baseline, dataset) in enumerate(tasks, 1):
        est_time = ESTIMATED_TIME.get((baseline, dataset), 10)
        print(f"  {i:2d}. {baseline:12s} - {dataset:12s} (~{est_time:2d}분)")
        task_queue.put((baseline, dataset))
    
    print(f"\n총 예상 시간 (순차): {sum(ESTIMATED_TIME.values())}분")
    print(f"병렬 실행 예상 시간: ~{sum(ESTIMATED_TIME.values()) / 4:.0f}분")
    print("="*80)
    
    # GPU 워커 시작
    workers = []
    for gpu_id in range(4):
        worker = GPUWorker(gpu_id, task_queue, log_dir)
        worker.start()
        workers.append(worker)
    
    # 모든 워커 완료 대기
    for worker in workers:
        worker.join()
    
    # 결과 출력
    print("\n" + "="*80)
    print("전체 작업 완료!")
    print("="*80)
    
    print("\nGPU별 완료 작업:")
    for worker in workers:
        print(f"\n[GPU {worker.gpu_id}] - {len(worker.completed_tasks)}개 완료")
        total_time = sum(t[2] for t in worker.completed_tasks)
        print(f"  총 실행 시간: {total_time:.1f}분")
        for baseline, dataset, elapsed in worker.completed_tasks:
            print(f"    - {baseline:12s} {dataset:12s} ({elapsed:.1f}분)")
    
    print("\n" + "="*80)
    print("결과 수집:")
    print("  python collect_results.py")
    print("="*80)

if __name__ == "__main__":
    main()
