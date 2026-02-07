#!/usr/bin/env python3
"""
실제 추론 시간 벤치마크 스크립트

논리적 earliness와 실제 시스템 시간을 함께 측정하여 비교합니다.

Usage:
    python benchmark_inference_time.py --dataset doore --fold 0
"""

import argparse
import time
import torch
import pickle
import numpy as np
from pathlib import Path

from baselines.calimera import CALIMERA
from baselines.EARLIEST.model import EARLIEST
from baselines.StopAndHop.model import StopAndHop
from data_preprocessing.data_preprocess import pad_sequences

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type='str', default='doore')
    parser.add_argument('--fold', type='int', default=0)
    parser.add_argument('--num_samples', type='int', default=100, help='Number of samples to benchmark')
    parser.add_argument('--warmup', type='int', default=5, help='Warmup iterations')
    parser.add_argument('--device', type='str', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_test_data(dataset, fold_idx):
    """테스트 데이터 로드"""
    fold_path = f'fold_data/{dataset}/fold_{fold_idx}.pkl'
    with open(fold_path, 'rb') as f:
        fold_data = pickle.load(f)
    
    test_data = fold_data['test_data']
    test_labels = fold_data['test_labels']
    
    return test_data, test_labels

def benchmark_calimera(test_data, test_labels, num_samples, warmup):
    """CALIMERA 추론 시간 측정"""
    print("\n" + "="*80)
    print("CALIMERA Benchmark")
    print("="*80)
    
    # 데이터 준비
    test_tensor, _ = pad_sequences(test_data[:num_samples], padding_type='mean')
    X_test = test_tensor.permute(0, 2, 1).numpy()
    y_test = test_labels[:num_samples].numpy() if torch.is_tensor(test_labels) else np.array(test_labels[:num_samples])
    
    # 모델 학습 (필요한 경우 - 실제로는 사전 학습된 모델 사용)
    print("Training CALIMERA (for benchmark)...")
    model = CALIMERA(delay_penalty=1)
    
    # 간단한 학습 데이터로 fit (실제로는 전체 데이터 사용)
    train_subset = min(50, len(test_data))
    train_tensor, _ = pad_sequences(test_data[:train_subset], padding_type='mean')
    X_train = train_tensor.permute(0, 2, 1).numpy()
    y_train = test_labels[:train_subset].numpy() if torch.is_tensor(test_labels) else np.array(test_labels[:train_subset])
    
    fit_start = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    print(f"Training time: {fit_time:.2f}s")
    
    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model.test(X_test[:5])
    
    # 실제 측정
    print(f"\nBenchmarking on {num_samples} samples...")
    inference_times = []
    
    for i in range(num_samples):
        X_single = X_test[i:i+1]
        
        start = time.perf_counter()
        stop_timestamps, y_pred = model.test(X_single)
        end = time.perf_counter()
        
        inference_times.append((end - start) * 1000)  # ms
    
    # 통계
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Mean inference time':<20} {mean_time:>12.2f} ms")
    print(f"{'Std deviation':<20} {std_time:>12.2f} ms")
    print(f"{'Median':<20} {median_time:>12.2f} ms")
    print(f"{'Min':<20} {min_time:>12.2f} ms")
    print(f"{'Max':<20} {max_time:>12.2f} ms")
    print(f"{'Throughput':<20} {1000/mean_time:>12.2f} samples/s")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'throughput': 1000/mean_time
    }

def benchmark_earliest(test_data, test_labels, num_samples, warmup, device):
    """EARLIEST 추론 시간 측정"""
    print("\n" + "="*80)
    print("EARLIEST Benchmark")
    print("="*80)
    
    # 데이터 준비
    test_tensor, _ = pad_sequences(test_data[:num_samples], padding_type='mean')
    test_tensor = test_tensor.to(device)
    
    # 모델 초기화
    input_channels = test_tensor.shape[2]
    num_classes = len(torch.unique(test_labels[:num_samples]))
    
    model = EARLIEST(
        ninp=input_channels,
        nhid=64,
        rnn_cell='LSTM',
        nlayers=1,
        nclasses=num_classes,
        lam=0.0
    ).to(device)
    model.eval()
    
    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            X_warmup = test_tensor[:5].permute(1, 0, 2)  # (T, B, V)
            _ = model(X_warmup, test=True)
    
    # 실제 측정
    print(f"\nBenchmarking on {num_samples} samples...")
    inference_times = []
    halt_points_list = []
    
    with torch.no_grad():
        for i in range(num_samples):
            X_single = test_tensor[i:i+1].permute(1, 0, 2)  # (T, 1, V)
            
            start = time.perf_counter()
            logits, _, halt_points = model(X_single, test=True)
            end = time.perf_counter()
            
            inference_times.append((end - start) * 1000)  # ms
            halt_points_list.append(halt_points.cpu().numpy())
    
    # 통계
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Earliness 계산
    max_T = test_tensor.shape[1]
    avg_halt = np.mean([np.mean(hp) for hp in halt_points_list])
    earliness = avg_halt / max_T
    
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Mean inference time':<20} {mean_time:>12.2f} ms")
    print(f"{'Std deviation':<20} {std_time:>12.2f} ms")
    print(f"{'Median':<20} {median_time:>12.2f} ms")
    print(f"{'Min':<20} {min_time:>12.2f} ms")
    print(f"{'Max':<20} {max_time:>12.2f} ms")
    print(f"{'Throughput':<20} {1000/mean_time:>12.2f} samples/s")
    print(f"{'Avg halt point':<20} {avg_halt:>12.2f}")
    print(f"{'Earliness':<20} {earliness*100:>12.2f} %")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'throughput': 1000/mean_time,
        'earliness': earliness
    }

def benchmark_stopandhop(test_data, test_labels, num_samples, warmup, device):
    """Stop and Hop 추론 시간 측정"""
    print("\n" + "="*80)
    print("Stop and Hop Benchmark")
    print("="*80)
    
    # 데이터 준비
    test_tensor, _ = pad_sequences(test_data[:num_samples], padding_type='mean')
    test_tensor = test_tensor.to(device)
    
    # 모델 초기화
    input_channels = test_tensor.shape[2]
    num_classes = len(torch.unique(test_labels[:num_samples]))
    
    config = {
        'n_epochs': 50,
        'nhid': 64,
        'rnn_cell': 'LSTM',
        'nlayers': 1,
    }
    
    model = StopAndHop(
        ninp=input_channels,
        nclasses=num_classes,
        config=config,
        std=0.1,
        lam=0.0
    ).to(device)
    model.eval()
    
    # Warmup
    print(f"\nWarmup ({warmup} iterations)...")
    with torch.no_grad():
        for _ in range(warmup):
            X_warmup = test_tensor[:5].numpy()
            _ = model(X_warmup, test=True)
    
    # 실제 측정
    print(f"\nBenchmarking on {num_samples} samples...")
    inference_times = []
    halt_points_list = []
    
    with torch.no_grad():
        for i in range(num_samples):
            X_single = test_tensor[i:i+1].numpy()
            
            start = time.perf_counter()
            logits, _, halt_points = model(X_single, test=True)
            end = time.perf_counter()
            
            inference_times.append((end - start) * 1000)  # ms
            halt_points_list.append(halt_points)
    
    # 통계
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    median_time = np.median(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Earliness 계산
    max_T = test_tensor.shape[1]
    avg_halt = np.mean(halt_points_list)
    earliness = avg_halt / max_T
    
    print(f"\n{'Metric':<20} {'Value':>15}")
    print("-" * 40)
    print(f"{'Mean inference time':<20} {mean_time:>12.2f} ms")
    print(f"{'Std deviation':<20} {std_time:>12.2f} ms")
    print(f"{'Median':<20} {median_time:>12.2f} ms")
    print(f"{'Min':<20} {min_time:>12.2f} ms")
    print(f"{'Max':<20} {max_time:>12.2f} ms")
    print(f"{'Throughput':<20} {1000/mean_time:>12.2f} samples/s")
    print(f"{'Avg halt point':<20} {avg_halt:>12.2f}")
    print(f"{'Earliness':<20} {earliness*100:>12.2f} %")
    
    return {
        'mean': mean_time,
        'std': std_time,
        'median': median_time,
        'min': min_time,
        'max': max_time,
        'throughput': 1000/mean_time,
        'earliness': earliness
    }

def main():
    args = setup_args()
    
    print("="*80)
    print("실제 추론 시간 벤치마크")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Num samples: {args.num_samples}")
    print(f"Device: {args.device}")
    print(f"Warmup iterations: {args.warmup}")
    
    # 데이터 로드
    print("\nLoading test data...")
    test_data, test_labels = load_test_data(args.dataset, args.fold)
    print(f"Loaded {len(test_data)} test samples")
    
    results = {}
    
    # CALIMERA 벤치마크
    try:
        results['calimera'] = benchmark_calimera(
            test_data, test_labels, 
            min(args.num_samples, len(test_data)), 
            args.warmup
        )
    except Exception as e:
        print(f"\n⚠️  CALIMERA benchmark failed: {e}")
    
    # EARLIEST 벤치마크
    try:
        results['earliest'] = benchmark_earliest(
            test_data, test_labels,
            min(args.num_samples, len(test_data)),
            args.warmup,
            args.device
        )
    except Exception as e:
        print(f"\n⚠️  EARLIEST benchmark failed: {e}")
    
    # Stop and Hop 벤치마크
    try:
        results['stopandhop'] = benchmark_stopandhop(
            test_data, test_labels,
            min(args.num_samples, len(test_data)),
            args.warmup,
            args.device
        )
    except Exception as e:
        print(f"\n⚠️  Stop and Hop benchmark failed: {e}")
    
    # 최종 비교
    print("\n" + "="*80)
    print("최종 비교")
    print("="*80)
    
    print(f"\n{'Baseline':<15} {'Mean (ms)':>12} {'Throughput':>15} {'Earliness (%)':>15}")
    print("-" * 80)
    
    for name, result in results.items():
        earliness_str = f"{result['earliness']*100:.2f}" if 'earliness' in result else "N/A"
        print(f"{name:<15} {result['mean']:>12.2f} {result['throughput']:>12.2f} sps {earliness_str:>15}")
    
    print("\n💡 해석:")
    print("  - Mean (ms): 평균 추론 시간 (낮을수록 좋음)")
    print("  - Throughput: 초당 처리 샘플 수 (높을수록 좋음)")
    print("  - Earliness: 논리적 중단 시점 (낮을수록 빠른 중단)")
    print()
    print("⚠️  주의: Earliness가 낮아도 실제 추론 시간이 느릴 수 있습니다!")

if __name__ == "__main__":
    main()
