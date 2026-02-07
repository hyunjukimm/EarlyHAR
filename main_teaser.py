import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from baselines.TEASER import TeaserEarlyClassifier, TeaserConfig
from data_preprocessing.data_preprocess import pad_sequences
import os
import yaml
import argparse
import logging
import time
import torch

def setup_args():
    parser = argparse.ArgumentParser()
    
    # General Arguments
    parser.add_argument('--dataset', type=str, default='casas')
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--padding', type=str, default='mean')
    
    # TEASER Specific
    parser.add_argument('--num_checkpoints', type=int, default=20, help='Number of checkpoints (S)')
    parser.add_argument('--slave_kind', type=str, default='minirocket', 
                        choices=['minirocket', 'muse', 'weasel'],
                        help='Slave classifier type')
    parser.add_argument('--svm_nu', type=float, default=0.05, help='One-class SVM nu parameter')
    parser.add_argument('--random_state', type=int, default=42)
    
    # Augmentation
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--aug_method', type=str, default='noise')
    
    # Quick test (가벼운 테스트용)
    parser.add_argument('--sample_ratio', type=float, default=None, 
                        help='Use only this fraction of data (e.g. 0.1 for 10%%) for quick test')
    
    return parser.parse_args()


def load_yaml_config(dataset_name):
    config_path = os.path.join('configs', f'{dataset_name}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML config not found: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = setup_args()
    
    # Setup logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'teaser_{args.dataset}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Starting TEASER training on {args.dataset} dataset")
    logging.info(f"Slave: {args.slave_kind}, Checkpoints: {args.num_checkpoints}")
    
    # Check sktime availability
    try:
        import sktime
        logging.info(f"sktime version: {sktime.__version__}")
    except ImportError:
        logging.error("sktime is not installed. Please install: pip install sktime")
        return
    
    # Results storage
    results_dir = f'results/{args.dataset}'
    os.makedirs(results_dir, exist_ok=True)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold_idx in range(args.k_fold):
        logging.info(f"\n{'='*50}")
        logging.info(f"Fold {fold_idx+1}/{args.k_fold}")
        logging.info(f"{'='*50}")
        
        # Load data
        fold_path = f'fold_data/{args.dataset}/fold_{fold_idx}.pkl'
        with open(fold_path, 'rb') as f:
            fold_data = pickle.load(f)
        
        train_data = fold_data['train_data']
        train_labels = fold_data['train_labels']
        test_data = fold_data['test_data']
        test_labels = fold_data['test_labels']
        
        # OpenPack: 30% stratified sampling (다른 베이스라인과 동일, seed=42)
        if args.dataset == 'openpack' and args.sample_ratio is None:
            SAMPLE_RATIO = 0.3
            openpack_seed = 42
            if len(train_data) > 1000:
                train_data, _, train_labels, _ = train_test_split(
                    train_data, train_labels, train_size=SAMPLE_RATIO,
                    stratify=train_labels, random_state=openpack_seed
                )
                logging.info(f"[OpenPack] Train 샘플링: {len(train_data)} samples ({SAMPLE_RATIO*100:.0f}%)")
            if len(test_data) > 1000:
                test_data, _, test_labels, _ = train_test_split(
                    test_data, test_labels, train_size=SAMPLE_RATIO,
                    stratify=test_labels, random_state=openpack_seed
                )
                logging.info(f"[OpenPack] Test 샘플링: {len(test_data)} samples ({SAMPLE_RATIO*100:.0f}%)")
        
        # Quick test: 데이터 10%만 사용
        if args.sample_ratio is not None:
            ratio = args.sample_ratio
            rng = np.random.RandomState(42)
            n_train = max(30, int(len(train_data) * ratio))
            n_test = max(20, int(len(test_data) * ratio))
            if len(train_data) > n_train:
                idx = rng.choice(len(train_data), n_train, replace=False)
                train_data = [train_data[i] for i in idx]
                train_labels = [train_labels[i] for i in idx]
                logging.info(f"[Quick test] Train 샘플링: {len(train_data)} samples ({ratio*100:.0f}%)")
            if len(test_data) > n_test:
                idx = rng.choice(len(test_data), n_test, replace=False)
                test_data = [test_data[i] for i in idx]
                test_labels = [test_labels[i] for i in idx]
                logging.info(f"[Quick test] Test 샘플링: {len(test_data)} samples ({ratio*100:.0f}%)")
        
        # Padding
        logging.info(f"[Padding] Type: {args.padding}")
        train_tensor, _ = pad_sequences(train_data, padding_type=args.padding)
        test_tensor, _ = pad_sequences(test_data, padding_type=args.padding)
        
        # Convert to numpy and transpose: (N, T, C) -> (N, C, T)
        X_train = train_tensor.numpy().transpose(0, 2, 1)
        X_test = test_tensor.numpy().transpose(0, 2, 1)
        y_train = np.array(train_labels)
        y_test = np.array(test_labels)
        
        logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # TEASER Config
        config = TeaserConfig(
            S=args.num_checkpoints,
            svm_nu=args.svm_nu,
            random_state=args.random_state
        )
        
        # Training
        logging.info(f"Training TEASER...")
        model = TeaserEarlyClassifier(config=config, slave_kind=args.slave_kind)
        
        train_start = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - train_start
        
        logging.info(f"Training completed in {training_time:.2f}s")
        logging.info(f"Optimal vote threshold: {model.threshold_}")
        
        # Inference with timing
        logging.info(f"Running inference...")
        
        # Warmup
        if len(X_test) > 0:
            _ = model.predict_with_earliness(X_test[:1])
        
        # Actual inference with timing
        inference_start = time.time()
        y_pred, earliness_vals = model.predict_with_earliness(X_test)
        inference_time = time.time() - inference_start
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        earliness = np.mean(earliness_vals)
        f_e = 2 * (1 - earliness) * f1 / ((1 - earliness) + f1 + 1e-8) if ((1 - earliness) + f1) > 0 else 0
        
        inference_time_ms = (inference_time / len(X_test)) * 1000  # ms per sample
        throughput_sps = len(X_test) / inference_time if inference_time > 0 else 0
        
        logging.info(f"\n[Test Results]")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"F1-Score (Macro): {f1:.4f}")
        logging.info(f"Earliness: {earliness:.4f}")
        logging.info(f"F-E Metric: {f_e:.4f}")
        logging.info(f"Training: {training_time:.2f}s")
        logging.info(f"Inference: {inference_time_ms:.2f}ms | Throughput: {throughput_sps:.1f} sps")
        
        # Store results
        fold_results.append({
            'fold': fold_idx,
            'accuracy': accuracy,
            'f1': f1,
            'earliness': earliness,
            'f_e': f_e,
            'inference_time_ms': inference_time_ms,
            'throughput_sps': throughput_sps,
            'training_time_sec': training_time,
            'vote_threshold': model.threshold_
        })
        
        all_predictions.extend(y_pred.tolist())
        all_true_labels.extend(y_test.tolist())
    
    # K-Fold Summary
    logging.info(f"\n{'='*50}")
    logging.info(f"K-Fold Cross-Validation Summary")
    logging.info(f"{'='*50}")
    
    df_results = pd.DataFrame(fold_results)
    
    for metric in ['accuracy', 'f1', 'earliness', 'f_e', 'inference_time_ms', 'throughput_sps', 'training_time_sec']:
        mean_val = df_results[metric].mean()
        std_val = df_results[metric].std()
        logging.info(f"{metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Confusion Matrix
    logging.info(f"\n{'='*50}")
    logging.info(f"Confusion Matrix (All Folds)")
    logging.info(f"{'='*50}")
    
    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    logging.info(f"\n{cm}")
    logging.info(f"\nNormalized Confusion Matrix:")
    logging.info(f"\n{cm_normalized}")
    
    # Save results
    df_results.to_csv(f'{results_dir}/teaser_per_fold_results.csv', index=False)
    
    summary_df = df_results[['accuracy', 'f1', 'earliness', 'f_e', 
                             'inference_time_ms', 'throughput_sps', 
                             'training_time_sec']].agg(['mean', 'std']).T
    summary_df.index.name = 'metric'
    summary_df.to_csv(f'{results_dir}/teaser_kfold_summary.csv')
    
    pd.DataFrame(cm).to_csv(f'{results_dir}/teaser_confusion_matrix.csv', index=False, header=False)
    pd.DataFrame(cm_normalized).to_csv(f'{results_dir}/teaser_confusion_matrix_normalized.csv', index=False, header=False)
    
    logging.info(f"\nResults saved to {results_dir}/")
    logging.info(f"  - teaser_kfold_summary.csv")
    logging.info(f"  - teaser_per_fold_results.csv")
    logging.info(f"  - teaser_confusion_matrix.csv")
    logging.info(f"  - teaser_confusion_matrix_normalized.csv")


if __name__ == '__main__':
    main()
