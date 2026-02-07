import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from baselines.calimera import CALIMERA
from data_preprocessing.data_preprocess import pad_sequences, balance_by_augmentation
import os
import yaml
import argparse
import gc

def load_yaml_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_args():
    parser = argparse.ArgumentParser()

    # --- General Arguments ---
    parser.add_argument('--dataset', type=str, default='doore')
    parser.add_argument('--padding', type=str, default='mean')
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--aug_method', type=str, default='noise')

    # --- CALIMERA Specific ---
    parser.add_argument('--delay_penalty', type=int, default=1)
    parser.add_argument('--k_fold', type=int, default=5)

    # --- Quick test ---
    parser.add_argument('--sample_ratio', type=float, default=None)
    parser.add_argument('--max_train', type=int, default=None)
    parser.add_argument('--max_test', type=int, default=None)

    # --- Parse ---
    args = parser.parse_args()

    # --- Dataset Config ---
    dataset_config = load_yaml_config(args.dataset)
    for key, value in dataset_config.items():
        setattr(args, key, value)

    # --- Paths ---
    args.save_dir = getattr(args, 'save_dir', 'save_model')
    os.makedirs(args.save_dir, exist_ok=True)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args

def f_e_metric(accuracy, earliness):
    return 2 * ((1 - earliness) * accuracy) / ((1 - earliness) + accuracy + 1e-8)

if __name__ == '__main__':

    args = setup_args()

    fold_metrics = {
        "accuracy": [],
        "f1": [],
        "earliness": [],
        "cost": [],
        "f_e": [],
        "inference_time_ms": [],
        "throughput_sps": [],
        "training_time_sec": []
    }
    per_fold_results = []
    
    # 전체 예측/레이블 저장 (Confusion Matrix용)
    all_predictions = []
    all_true_labels = []

    for fold_idx in range(args.k_fold):
        print(f"\n=== Fold {fold_idx} ===")

        # Load data
        fold_path = f'fold_data/{args.dataset}/fold_{fold_idx}.pkl'
        with open(fold_path, 'rb') as f:
            fold_data = pickle.load(f)

        train_data = fold_data['train_data']
        train_labels = fold_data['train_labels']
        test_data = fold_data['test_data']
        test_labels = fold_data['test_labels']

        # Augmentation
        if args.augment:
            print(f"[Augment] Method: {args.aug_method}")
            train_data, train_labels = balance_by_augmentation(train_data, train_labels, method=args.aug_method)

        # Quick test: sample_ratio or max_train/max_test
        rng = np.random.RandomState(42)
        if args.sample_ratio is not None:
            n_train = max(30, int(len(train_data) * args.sample_ratio))
            n_test = max(20, int(len(test_data) * args.sample_ratio))
            if len(train_data) > n_train:
                idx = rng.choice(len(train_data), n_train, replace=False)
                train_data = [train_data[i] for i in idx]
                train_labels = np.array(train_labels)[idx]
                print(f"[Quick test] Train: {len(train_data)} ({args.sample_ratio*100:.0f}%)")
            if len(test_data) > n_test:
                idx = rng.choice(len(test_data), n_test, replace=False)
                test_data = [test_data[i] for i in idx]
                test_labels = np.array(test_labels)[idx]
                print(f"[Quick test] Test: {len(test_data)} ({args.sample_ratio*100:.0f}%)")
        if args.max_train is not None and len(train_data) > args.max_train:
            idx = rng.choice(len(train_data), args.max_train, replace=False)
            train_data = [train_data[i] for i in idx]
            train_labels = np.array(train_labels)[idx]
        if args.max_test is not None and len(test_data) > args.max_test:
            idx = rng.choice(len(test_data), args.max_test, replace=False)
            test_data = [test_data[i] for i in idx]
            test_labels = np.array(test_labels)[idx]
        
        # OpenPack: 샘플링 (시간/메모리 최적화) - skip if quick test
        if args.dataset == 'openpack' and args.sample_ratio is None:
            SAMPLE_RATIO = 0.3  # 30%만 사용 (13,950 → ~4,185)
            
            # Train 샘플링 (Stratified)
            if len(train_data) > 1000:
                from sklearn.model_selection import train_test_split
                train_data, _, train_labels, _ = train_test_split(
                    train_data, train_labels, 
                    train_size=SAMPLE_RATIO, 
                    stratify=train_labels, 
                    random_state=42
                )
                print(f"[OpenPack] Train 샘플링: {len(train_data)} samples ({SAMPLE_RATIO*100:.0f}%)")
            
            # Test 샘플링 (Stratified)
            if len(test_data) > 1000:
                test_data, _, test_labels, _ = train_test_split(
                    test_data, test_labels, 
                    train_size=SAMPLE_RATIO, 
                    stratify=test_labels, 
                    random_state=42
                )
                print(f"[OpenPack] Test 샘플링: {len(test_data)} samples ({SAMPLE_RATIO*100:.0f}%)")

        # Padding
        train_tensor, _ = pad_sequences(train_data, padding_type=args.padding)
        test_tensor, _ = pad_sequences(test_data, padding_type=args.padding)

        # In-place 변환 및 메모리 해제
        X_train = train_tensor.permute(0, 2, 1).contiguous().numpy()
        X_test = test_tensor.permute(0, 2, 1).contiguous().numpy()
        
        y_train = train_labels.numpy() if isinstance(train_labels, torch.Tensor) else np.array(train_labels)
        y_test = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else np.array(test_labels)
        
        # 중간 변수 메모리 해제
        del train_tensor, test_tensor, train_data, test_data, train_labels, test_labels
        gc.collect()  # 가비지 컬렉션 강제 실행

        # Fit and Evaluate CALIMERA
        import time
        model = CALIMERA(delay_penalty=args.delay_penalty)
        
        # 학습 시간 측정
        train_start = time.perf_counter()
        model.fit(X_train, y_train)
        train_end = time.perf_counter()
        training_time = train_end - train_start
        
        # 추론 시간 측정
        import time
        inference_times = []
        fold_predictions = []
        fold_timestamps = []
        
        for i in range(len(X_test)):
            X_single = X_test[i:i+1]
            start = time.perf_counter()
            stop_timestamps_single, y_pred_single = model.test(X_single)
            end = time.perf_counter()
            
            inference_times.append((end - start) * 1000)  # ms
            fold_predictions.extend(y_pred_single)
            fold_timestamps.extend(stop_timestamps_single)
        
        y_pred = np.array(fold_predictions)
        stop_timestamps = fold_timestamps
        
        # 전체 결과 저장 (Confusion Matrix용)
        all_predictions.extend(y_pred.tolist())
        all_true_labels.extend(y_test.tolist())
        
        # 평가 지표
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
        cost = 1.0 - accuracy + args.delay_penalty * earliness
        f_e = f_e_metric(accuracy, earliness)
        
        # 추론 시간 통계
        mean_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        throughput = 1000 / mean_inference_time

        print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Earliness: {earliness:.4f} | Cost: {cost:.4f} | F-E: {f_e:.4f}")
        print(f"Training: {training_time:.2f}s | Inference: {mean_inference_time:.2f}±{std_inference_time:.2f}ms | Throughput: {throughput:.1f} sps")

        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["f1"].append(f1)
        fold_metrics["earliness"].append(earliness)
        fold_metrics["cost"].append(cost)
        fold_metrics["f_e"].append(f_e)
        fold_metrics["inference_time_ms"].append(mean_inference_time)
        fold_metrics["throughput_sps"].append(throughput)
        fold_metrics["training_time_sec"].append(training_time)

        per_fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "f1": f1,
            "earliness": earliness,
            "cost": cost,
            "f_e": f_e,
            "inference_time_ms": mean_inference_time,
            "throughput_sps": throughput,
            "training_time_sec": training_time
        })
        
        # Fold 종료 후 메모리 정리
        del X_train, X_test, y_train, y_test, y_pred, stop_timestamps, inference_times, model
        gc.collect()

    # Summary
    print("\n=== K-Fold Summary ===")
    for key in fold_metrics:
        values = np.array(fold_metrics[key])
        print(f"{key.upper()}: {values.mean():.4f} ± {values.std():.4f}")
    
    # Confusion Matrix (전체 결과)
    print("\n=== Confusion Matrix (All Folds) ===")
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(cm)
    
    # Normalized Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    print("\n=== Normalized Confusion Matrix ===")
    print(cm_normalized)

    # Save results
    result_dir = f"results/{args.dataset}"
    os.makedirs(result_dir, exist_ok=True)
    
    # Save summary CSV
    summary_data = {
        "metric": list(fold_metrics.keys()),
        "mean": [np.mean(fold_metrics[k]) for k in fold_metrics.keys()],
        "std": [np.std(fold_metrics[k]) for k in fold_metrics.keys()]
    }
    pd.DataFrame(summary_data).to_csv(f"{result_dir}/calimera_kfold_summary.csv", index=False)
    
    # Save per-fold results
    pd.DataFrame(per_fold_results).to_csv(f"{result_dir}/calimera_per_fold_results.csv", index=False)
    
    # Save Confusion Matrix
    pd.DataFrame(cm).to_csv(f"{result_dir}/calimera_confusion_matrix.csv", index=False)
    pd.DataFrame(cm_normalized).to_csv(f"{result_dir}/calimera_confusion_matrix_normalized.csv", index=False)
    
    print(f"\nResults saved to {result_dir}/")
    print(f"  - calimera_kfold_summary.csv")
    print(f"  - calimera_per_fold_results.csv")
    print(f"  - calimera_confusion_matrix.csv")
    print(f"  - calimera_confusion_matrix_normalized.csv")
