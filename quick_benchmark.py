#!/usr/bin/env python3
"""
간단한 추론 시간 벤치마크
"""

import time
import torch
import pickle
import numpy as np
import sys

# 데이터 로드
print("="*80)
print("간단한 추론 시간 벤치마크")
print("="*80)

dataset = sys.argv[1] if len(sys.argv) > 1 else 'doore'
fold_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"Dataset: {dataset}")
print(f"Fold: {fold_idx}")

# 데이터 로드
fold_path = f'fold_data/{dataset}/fold_{fold_idx}.pkl'
print(f"\n데이터 로드: {fold_path}")

with open(fold_path, 'rb') as f:
    fold_data = pickle.load(f)

test_data = fold_data['test_data'][:10]  # 10개만 테스트
test_labels = fold_data['test_labels'][:10]

print(f"테스트 샘플: {len(test_data)}개")

# 벤치마크 함수
def benchmark_model(model_name, forward_func, test_data, warmup=3):
    print(f"\n{'='*80}")
    print(f"{model_name} 벤치마크")
    print(f"{'='*80}")
    
    # Warmup
    print(f"Warmup ({warmup}회)...")
    for _ in range(warmup):
        try:
            _ = forward_func(test_data[0])
        except:
            pass
    
    # 측정
    print(f"측정 중 ({len(test_data)}개 샘플)...")
    times = []
    
    for i, sample in enumerate(test_data):
        start = time.perf_counter()
        try:
            _ = forward_func(sample)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        except Exception as e:
            print(f"  샘플 {i} 에러: {e}")
            continue
    
    if times:
        mean_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)
        
        print(f"\n결과:")
        print(f"  평균 추론 시간: {mean_time:.2f} ms")
        print(f"  표준편차:      {std_time:.2f} ms")
        print(f"  중앙값:        {median_time:.2f} ms")
        print(f"  처리량:        {1000/mean_time:.2f} samples/s")
        
        return mean_time
    else:
        print("측정 실패!")
        return None

# CALIMERA 테스트
try:
    from baselines.calimera import CALIMERA
    from data_preprocessing.data_preprocess import pad_sequences
    
    print("\n" + "="*80)
    print("CALIMERA 준비 중...")
    print("="*80)
    
    # 간단한 학습
    train_tensor, _ = pad_sequences(test_data[:5], padding_type='mean')
    X_train = train_tensor.permute(0, 2, 1).numpy()
    y_train = test_labels[:5].numpy() if torch.is_tensor(test_labels) else np.array(test_labels[:5])
    
    model_cal = CALIMERA(delay_penalty=1)
    model_cal.fit(X_train, y_train)
    
    def calimera_forward(sample):
        tensor, _ = pad_sequences([sample], padding_type='mean')
        X = tensor.permute(0, 2, 1).numpy()
        return model_cal.test(X)
    
    cal_time = benchmark_model("CALIMERA", calimera_forward, test_data[5:], warmup=2)
    
except Exception as e:
    print(f"\n⚠️  CALIMERA 벤치마크 실패: {e}")
    cal_time = None

# EARLIEST 테스트
try:
    from baselines.EARLIEST.model import EARLIEST
    
    print("\n" + "="*80)
    print("EARLIEST 준비 중...")
    print("="*80)
    
    # 모델 초기화
    test_tensor, _ = pad_sequences(test_data, padding_type='mean')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_channels = test_tensor.shape[2]
    num_classes = len(torch.unique(test_labels))
    
    model_earliest = EARLIEST(
        ninp=input_channels,
        nhid=64,
        rnn_cell='LSTM',
        nlayers=1,
        nclasses=num_classes,
        lam=0.0
    ).to(device)
    model_earliest.eval()
    
    def earliest_forward(sample):
        tensor, _ = pad_sequences([sample], padding_type='mean')
        tensor = tensor.to(device)
        X = tensor.permute(1, 0, 2)  # (T, 1, V)
        with torch.no_grad():
            return model_earliest(X, test=True)
    
    ear_time = benchmark_model("EARLIEST", earliest_forward, test_data, warmup=2)
    
except Exception as e:
    print(f"\n⚠️  EARLIEST 벤치마크 실패: {e}")
    ear_time = None

# Stop and Hop 테스트
try:
    from baselines.StopAndHop.model import StopAndHop
    
    print("\n" + "="*80)
    print("Stop and Hop 준비 중...")
    print("="*80)
    
    config = {
        'n_epochs': 50,
        'nhid': 64,
        'rnn_cell': 'LSTM',
        'nlayers': 1,
    }
    
    model_sh = StopAndHop(
        ninp=input_channels,
        nclasses=num_classes,
        config=config,
        std=0.1,
        lam=0.0
    ).to(device)
    model_sh.eval()
    
    def stopandhop_forward(sample):
        tensor, _ = pad_sequences([sample], padding_type='mean')
        X = tensor.numpy()
        with torch.no_grad():
            return model_sh(X, test=True)
    
    sh_time = benchmark_model("Stop and Hop", stopandhop_forward, test_data, warmup=2)
    
except Exception as e:
    print(f"\n⚠️  Stop and Hop 벤치마크 실패: {e}")
    sh_time = None

# 최종 비교
print("\n" + "="*80)
print("최종 비교")
print("="*80)

results = []
if cal_time:
    results.append(("CALIMERA", cal_time))
if ear_time:
    results.append(("EARLIEST", ear_time))
if sh_time:
    results.append(("Stop and Hop", sh_time))

if results:
    results.sort(key=lambda x: x[1])
    
    print(f"\n{'순위':<5} {'Baseline':<15} {'추론 시간':<15} {'처리량':<15}")
    print("-" * 55)
    
    for i, (name, time_ms) in enumerate(results, 1):
        throughput = 1000 / time_ms
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"{emoji} {i:<3} {name:<15} {time_ms:>10.2f} ms {throughput:>10.2f} sps")
    
    print("\n💡 해석:")
    print(f"  - 가장 빠른 모델: {results[0][0]} ({results[0][1]:.2f} ms)")
    print(f"  - 가장 느린 모델: {results[-1][0]} ({results[-1][1]:.2f} ms)")
    print(f"  - 속도 차이: {results[-1][1] / results[0][1]:.1f}배")
else:
    print("⚠️  측정 결과가 없습니다.")

print()
