"""CALIMERA 병목 분석 스크립트"""
import numpy as np
import pickle
import time
import torch
import sys
sys.path.insert(0, '/home/juice/EarlyHAR/EarlyHAR')

from baselines.calimera import CALIMERA
from data.dataset import pad_sequences

# 작은 데이터셋으로 테스트
print("=" * 60)
print("CALIMERA 병목 분석 시작")
print("=" * 60)

# Load doore fold 0
fold_path = 'fold_data/doore/fold_0.pkl'
with open(fold_path, 'rb') as f:
    fold_data = pickle.load(f)

train_data = fold_data['train_data']
train_labels = fold_data['train_labels']

# Padding
train_tensor, _ = pad_sequences(train_data, padding_type='zero')
X_train = train_tensor.permute(0, 2, 1).numpy()
y_train = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else np.array(train_labels)

print(f"\n데이터 크기:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  Max Timestamp: {X_train.shape[-1]}")

# CALIMERA 학습 with 프로파일링
model = CALIMERA(delay_penalty=0.001)

print("\n" + "=" * 60)
print("단계별 학습 시간 측정")
print("=" * 60)

# 1. Generate Timestamps
start = time.perf_counter()
timestamps = CALIMERA._generate_timestamps(max_timestamp=X_train.shape[-1])
t1 = time.perf_counter() - start
print(f"\n[1] Timestamp 생성: {t1:.4f}s")
print(f"    - 생성된 timestamps: {len(timestamps)}개")
print(f"    - Timestamps: {timestamps[:5]}... {timestamps[-3:]}")

# 2. Learn Feature Extractors
print(f"\n[2] Feature Extractor 학습 (MiniRocket):")
start_total = time.perf_counter()
extractors = []
for i, timestamp in enumerate(timestamps):
    start = time.perf_counter()
    if timestamp < 9:
        extractors.append(lambda x: x.reshape(x.shape[0], -1))
        elapsed = time.perf_counter() - start
        print(f"    - [{i+1:2d}/{len(timestamps)}] t={timestamp:3d}: {elapsed:.4f}s (Flatten)")
    else:
        X_sub = X_train[:, :, :timestamp]
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        extractor = MiniRocketMultivariate()
        extractor.fit(X_sub)
        extractors.append(extractor.transform)
        elapsed = time.perf_counter() - start
        print(f"    - [{i+1:2d}/{len(timestamps)}] t={timestamp:3d}: {elapsed:.4f}s (MiniRocket)")
t2 = time.perf_counter() - start_total
print(f"  총 Feature Extractor 학습 시간: {t2:.4f}s")

# 3. Get Features
print(f"\n[3] Feature 추출:")
start = time.perf_counter()
features = CALIMERA._get_features(X_train, extractors, timestamps)
t3 = time.perf_counter() - start
print(f"    - 소요 시간: {t3:.4f}s")
print(f"    - Feature shapes: {[f.shape for f in features[:3]]}...")

# 4. Learn Classifiers
print(f"\n[4] Classifier 학습 (RidgeCV + Calibration):")
start_total = time.perf_counter()
from baselines.calimera import WeakClassifier
classifiers = []
for t in range(len(timestamps)):
    start = time.perf_counter()
    classifier = WeakClassifier()
    classifier.fit(features[t], y_train)
    classifiers.append(classifier)
    elapsed = time.perf_counter() - start
    print(f"    - [{t+1:2d}/{len(timestamps)}] t={timestamps[t]:3d}: {elapsed:.4f}s")
t4 = time.perf_counter() - start_total
print(f"  총 Classifier 학습 시간: {t4:.4f}s")

# 5. Stopping Module
print(f"\n[5] Stopping Module 학습:")
start = time.perf_counter()
predictors = []
costs = []
for classifier in classifiers:
    costs.append(classifier.costs_for_training_stopping_module)
    predictors.append([
        np.argmax(s) for s in classifier.predictors_for_training_stopping_module
    ])
predictors = np.asarray(predictors)
costs = np.asarray(costs)

from baselines.calimera import StoppingModule, KernelRidgeRegressionWrapper
stopping_module = StoppingModule()
stopping_module.fit(
    predictors,
    costs,
    timestamps,
    model.delay_penalty,
    KernelRidgeRegressionWrapper
)
t5 = time.perf_counter() - start
print(f"    - 소요 시간: {t5:.4f}s")

# 총합
total_time = t1 + t2 + t3 + t4 + t5
print("\n" + "=" * 60)
print("총 학습 시간 분석")
print("=" * 60)
print(f"[1] Timestamp 생성:         {t1:8.4f}s ({t1/total_time*100:5.1f}%)")
print(f"[2] MiniRocket 학습:        {t2:8.4f}s ({t2/total_time*100:5.1f}%) ⚠️")
print(f"[3] Feature 추출:           {t3:8.4f}s ({t3/total_time*100:5.1f}%)")
print(f"[4] Classifier 학습:        {t4:8.4f}s ({t4/total_time*100:5.1f}%) ⚠️")
print(f"[5] Stopping Module 학습:   {t5:8.4f}s ({t5/total_time*100:5.1f}%)")
print("-" * 60)
print(f"총 학습 시간:               {total_time:8.4f}s")
print("=" * 60)

print("\n주요 병목:")
bottlenecks = [
    ("MiniRocket 학습", t2, t2/total_time*100),
    ("Classifier 학습", t4, t4/total_time*100),
    ("Stopping Module", t5, t5/total_time*100),
]
bottlenecks.sort(key=lambda x: x[1], reverse=True)
for name, time_val, pct in bottlenecks:
    if pct > 10:
        print(f"  🔴 {name}: {time_val:.2f}s ({pct:.1f}%)")
    elif pct > 5:
        print(f"  🟡 {name}: {time_val:.2f}s ({pct:.1f}%)")

print("\n개선 방안:")
print("  1. NUM_TIMESTAMPS 줄이기 (20 → 10): 학습 시간 50% 감소")
print("  2. MiniRocket 대신 간단한 feature 사용: 80% 감소")
print("  3. RidgeCV alpha 수 줄이기 (10 → 5): 20% 감소")
print("  4. GPU 기반 feature extractor: 2-3배 속도 향상")
