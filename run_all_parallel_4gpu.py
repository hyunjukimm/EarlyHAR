#!/usr/bin/env python3
"""
4개 GPU로 모든 Baseline 병렬 실행 (동적 스케줄링)
- 5개 데이터셋 × 3개 Baseline = 15개 작업
- GPU 4개에 동적으로 할당
- 가장 빠른 실행을 위한 최적화
"""

import subprocess
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

# 실험 정의
DATASETS = ["aras", "casas", "doore", "openpack", "opportunity"]
BASELINES = ["calimera", "earliest", "stopandhop"]

# 작업 생성 (우선순위: 빠른 작업 먼저)
def create_tasks():
    """작업 우선순위 설정 (빠른 것 먼저)"""
    tasks = []
    
    # Priority 1: EARLIEST (가장 빠름)
    for dataset in DATASETS:
        tasks.append(("earliest", dataset, 1))
    
    # Priority 2: CALIMERA (중간)
    for dataset in DATASETS:
        tasks.append(("calimera", dataset, 2))
    
    # Priority 3: Stop and Hop (느림)
    for dataset in DATASETS:
        tasks.append(("stopandhop", dataset, 3))
    
    # 우선순위 순으로 정렬
    tasks.sort(key=lambda x: x[2])
    return tasks

def get_command(baseline, dataset):
    """각 baseline별 실행 명령어 생성"""
    base_dir = "/home/juice/EarlyHAR/EarlyHAR"
    conda_cmd = "source ~/miniconda3/etc/profile.d/conda.sh && conda activate pytorch-env && "
    conda_cmd += f"export NUMBA_CACHE_DIR={base_dir}/.numba_cache && "
    
    if baseline == "calimera":
        cmd = conda_cmd + f"python {base_dir}/main_cal.py "
        cmd += f"--dataset {dataset} --k_fold 5 --padding mean --augment True --aug_method noise --delay_penalty 1"
        
    elif baseline == "earliest":
        cmd = conda_cmd + f"python {base_dir}/main_earliest.py "
        cmd += f"--dataset {dataset} --k_fold 5 --padding mean --augment --aug_method noise "
        cmd += f"--epochs 50 --batch_size 32 --nhid 64 --patience 10"
        
    elif baseline == "stopandhop":
        cmd = conda_cmd + f"python {base_dir}/main_stopandhop.py "
        cmd += f"--dataset {dataset} --k_fold 5 --padding mean --augment --aug_method noise "
        cmd += f"--n_epochs 50 --batch_size 32 --nhid 64 --patience 10"
    
    return cmd

class GPUWorker(threading.Thread):
    """GPU별 작업 처리 스레드"""
    
    def __init__(self, gpu_id, task_queue, results):
        super().__init__()
        self.gpu_id = gpu_id
        self.task_queue = task_queue
        self.results = results
        self.daemon = True
        
    def run(self):
        while True:
            try:
                baseline, dataset, priority = self.task_queue.get(timeout=1)
            except queue.Empty:
                break
                
            task_name = f"{baseline}_{dataset}"
            log_file = f"logs/{task_name}_gpu{self.gpu_id}.log"
            
            print(f"[GPU {self.gpu_id}] 🟢 시작: {baseline.upper()} on {dataset}")
            start_time = time.time()
            
            # 명령어 실행
            cmd = get_command(baseline, dataset)
            env = {"CUDA_VISIBLE_DEVICES": str(self.gpu_id)}
            
            try:
                with open(log_file, 'w') as f:
                    process = subprocess.run(
                        cmd,
                        shell=True,
                        executable='/bin/bash',
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        env={**subprocess.os.environ.copy(), **env}
                    )
                
                elapsed = time.time() - start_time
                
                if process.returncode == 0:
                    status_emoji = "✅"
                    status_text = "완료"
                else:
                    status_emoji = "❌"
                    status_text = "실패"
                
                result = {
                    'gpu': self.gpu_id,
                    'baseline': baseline,
                    'dataset': dataset,
                    'status': 'success' if process.returncode == 0 else 'failed',
                    'elapsed': elapsed,
                    'log': log_file
                }
                
                self.results.append(result)
                
                print(f"[GPU {self.gpu_id}] {status_emoji} {status_text}: {baseline.upper()} on {dataset} ({elapsed/60:.1f}분)")
                
            except Exception as e:
                print(f"[GPU {self.gpu_id}] ❌ 에러: {baseline} on {dataset} - {e}")
                self.results.append({
                    'gpu': self.gpu_id,
                    'baseline': baseline,
                    'dataset': dataset,
                    'status': 'error',
                    'error': str(e),
                    'log': log_file
                })
            
            self.task_queue.task_done()

def main():
    print("=" * 70)
    print("🚀 4개 GPU로 전체 실험 병렬 실행")
    print("=" * 70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 로그 디렉토리 생성
    Path("logs").mkdir(exist_ok=True)
    
    # 작업 큐 생성
    task_queue = queue.Queue()
    tasks = create_tasks()
    
    print(f"📊 총 작업 수: {len(tasks)}개")
    print(f"   - EARLIEST: {sum(1 for t in tasks if t[0]=='earliest')}개")
    print(f"   - CALIMERA: {sum(1 for t in tasks if t[0]=='calimera')}개")
    print(f"   - Stop & Hop: {sum(1 for t in tasks if t[0]=='stopandhop')}개")
    print()
    
    for task in tasks:
        task_queue.put(task)
    
    # GPU 워커 시작
    results = []
    workers = []
    
    print("🔧 GPU 워커 시작...")
    for gpu_id in range(4):  # GPU 0, 1, 2, 3
        worker = GPUWorker(gpu_id, task_queue, results)
        worker.start()
        workers.append(worker)
        print(f"   - GPU {gpu_id}: 준비 완료")
    
    print()
    print("=" * 70)
    print("⏳ 실험 진행 중... (Ctrl+C로 중단 가능)")
    print("=" * 70)
    print()
    
    start_time = time.time()
    
    # 모든 작업 완료 대기
    task_queue.join()
    
    # 워커 종료 대기
    for worker in workers:
        worker.join()
    
    total_elapsed = time.time() - start_time
    
    # 결과 출력
    print()
    print("=" * 70)
    print("✅ 전체 실험 완료!")
    print("=" * 70)
    print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 소요 시간: {total_elapsed/3600:.2f}시간 ({total_elapsed/60:.1f}분)")
    print()
    
    # 성공/실패 통계
    success = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - success
    
    print(f"📊 실험 결과:")
    print(f"   - 성공: {success}개")
    print(f"   - 실패: {failed}개")
    print(f"   - 총: {len(results)}개")
    print()
    
    # Baseline별 통계
    print("📈 Baseline별 평균 시간:")
    for baseline in BASELINES:
        baseline_results = [r for r in results if r['baseline'] == baseline and r['status'] == 'success']
        if baseline_results:
            avg_time = sum(r['elapsed'] for r in baseline_results) / len(baseline_results)
            print(f"   - {baseline.upper()}: {avg_time/60:.1f}분")
    print()
    
    # GPU별 통계
    print("🖥️  GPU별 처리 작업:")
    for gpu_id in range(4):
        gpu_results = [r for r in results if r['gpu'] == gpu_id]
        print(f"   - GPU {gpu_id}: {len(gpu_results)}개 작업")
    print()
    
    # 실패한 작업 출력
    if failed > 0:
        print("❌ 실패한 작업:")
        for r in results:
            if r['status'] != 'success':
                print(f"   - {r['baseline']} on {r['dataset']} (GPU {r['gpu']})")
                print(f"     로그: {r['log']}")
        print()
    
    print("📁 결과 위치:")
    for dataset in DATASETS:
        print(f"   - results/{dataset}/")
    print()
    print("📊 결과 수집 명령:")
    print("   python collect_results.py")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()
