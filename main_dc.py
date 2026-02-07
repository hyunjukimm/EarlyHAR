"""
Divide-and-Conquer (ECM) baseline - same pipeline as other baselines.
Loads fold data, converts (T,C) to MTDSample (each channel = component),
trains ECM, evaluates with earliness, saves same CSV/metrics.
"""
import argparse
import importlib.util
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from data_preprocessing.data_preprocess import pad_sequences


def _load_dc():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "baselines", "Divide-and-Conquer", "divideconquer.py")
    spec = importlib.util.spec_from_file_location("divideconquer", path)
    dc = importlib.util.module_from_spec(spec)
    sys.modules["divideconquer"] = dc
    spec.loader.exec_module(dc)
    return dc


def setup_args():
    parser = argparse.ArgumentParser(description="Divide-and-Conquer (ECM) baseline")
    parser.add_argument("--dataset", type=str, default="casas")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--padding", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--alpha", type=float, default=0.9, help="MRD alpha threshold")
    parser.add_argument(
        "--n_component_groups",
        type=int,
        default=None,
        help="Group channels into K components (average within group). Reduces GP/MRD cost.",
    )
    parser.add_argument("--use_clustering", action="store_true", help="Use clustering in MRD estimation")
    parser.add_argument("--sample_ratio", type=float, default=None, help="Quick test: use this fraction of data")
    parser.add_argument("--max_train", type=int, default=None, help="Cap train size")
    parser.add_argument("--max_test", type=int, default=None, help="Cap test size")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def group_channels(X: np.ndarray, K: int) -> np.ndarray:
    """X: (N, T, C). Group C channels into K groups by averaging. Returns (N, T, K)."""
    N, T, C = X.shape
    K = min(max(1, K), C)
    out = np.zeros((N, T, K), dtype=X.dtype)
    for g in range(K):
        start = (g * C) // K
        end = ((g + 1) * C) // K
        if end > start:
            out[:, :, g] = X[:, :, start:end].mean(axis=2)
        else:
            out[:, :, g] = X[:, :, start]
    return out


def arrays_to_mtd_samples(X: np.ndarray, y: np.ndarray) -> list:
    """X: (N, T, C). Each channel = component => MTDSample with C components."""
    dc = _load_dc()
    samples = []
    for i in range(X.shape[0]):
        comps = [X[i, :, j].astype(np.float64) for j in range(X.shape[2])]
        samples.append(dc.MTDSample(components=comps, label=int(y[i])))
    return samples


def main():
    args = setup_args()
    np.random.seed(args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = os.path.join("logs", f"dc_{args.dataset}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    dc = _load_dc()
    results_dir = os.path.join("results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)

    fold_results = []
    all_predictions = []
    all_true_labels = []

    for fold_idx in range(args.k_fold):
        logging.info("\n" + "=" * 50)
        logging.info(f"Fold {fold_idx + 1}/{args.k_fold}")
        logging.info("=" * 50)

        fold_path = os.path.join("fold_data", args.dataset, f"fold_{fold_idx}.pkl")
        if not os.path.exists(fold_path):
            logging.error(f"Fold file not found: {fold_path}")
            break

        with open(fold_path, "rb") as f:
            fold_data = pickle.load(f)

        train_data = fold_data["train_data"]
        train_labels = np.array(fold_data["train_labels"])
        test_data = fold_data["test_data"]
        test_labels = np.array(fold_data["test_labels"])

        if args.dataset == "openpack" and args.sample_ratio is None:
            from sklearn.model_selection import train_test_split
            SAMPLE_RATIO = 0.3
            if len(train_data) > 1000:
                train_data, _, train_labels, _ = train_test_split(
                    train_data, train_labels, train_size=SAMPLE_RATIO, stratify=train_labels, random_state=args.seed
                )
                logging.info(f"[OpenPack] Train 샘플링: {len(train_data)} samples ({SAMPLE_RATIO*100:.0f}%)")
            if len(test_data) > 1000:
                test_data, _, test_labels, _ = train_test_split(
                    test_data, test_labels, train_size=SAMPLE_RATIO, stratify=test_labels, random_state=args.seed
                )
                logging.info(f"[OpenPack] Test 샘플링: {len(test_data)} samples ({SAMPLE_RATIO*100:.0f}%)")

        if args.sample_ratio is not None:
            rng = np.random.RandomState(args.seed)
            n_train = max(30, int(len(train_data) * args.sample_ratio))
            n_test = max(20, int(len(test_data) * args.sample_ratio))
            if len(train_data) > n_train:
                idx = rng.choice(len(train_data), n_train, replace=False)
                train_data = [train_data[i] for i in idx]
                train_labels = train_labels[idx]
                logging.info(f"[Quick test] Train 샘플링: {len(train_data)} ({args.sample_ratio*100:.0f}%)")
            if len(test_data) > n_test:
                idx = rng.choice(len(test_data), n_test, replace=False)
                test_data = [test_data[i] for i in idx]
                test_labels = test_labels[idx]
                logging.info(f"[Quick test] Test 샘플링: {len(test_data)} ({args.sample_ratio*100:.0f}%)")

        if args.max_train is not None and len(train_data) > args.max_train:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(train_data), args.max_train, replace=False)
            train_data = [train_data[i] for i in idx]
            train_labels = train_labels[idx]
            logging.info(f"[Smoke test] Train capped: {len(train_data)}")
        if args.max_test is not None and len(test_data) > args.max_test:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(test_data), args.max_test, replace=False)
            test_data = [test_data[i] for i in idx]
            test_labels = test_labels[idx]
            logging.info(f"[Smoke test] Test capped: {len(test_data)}")

        logging.info(f"[Padding] Type: {args.padding}")
        train_padded, _ = pad_sequences(train_data, padding_type=args.padding)
        test_padded, _ = pad_sequences(test_data, padding_type=args.padding)

        if hasattr(train_padded, "cpu"):
            X_train = train_padded.cpu().numpy()
            X_test = test_padded.cpu().numpy()
        else:
            X_train = np.asarray(train_padded)
            X_test = np.asarray(test_padded)

        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]

        T = X_train.shape[1]
        C = X_train.shape[2]
        n_classes = len(set(train_labels) | set(test_labels))

        if getattr(args, "n_component_groups", None) is not None and args.n_component_groups < C:
            K = args.n_component_groups
            X_train = group_channels(X_train, K)
            X_test = group_channels(X_test, K)
            logging.info(f"채널 그룹화: C={C} -> K={K} components (평균으로 묶음)")

        n_components = X_train.shape[2]
        lambdas = np.ones(n_components, dtype=float)

        train_samples = arrays_to_mtd_samples(X_train, train_labels)
        test_samples = arrays_to_mtd_samples(X_test, test_labels)

        logging.info(f"Train: {len(train_samples)} samples, Test: {len(test_samples)}, T={T}, components={n_components}")

        logging.info("Training ECM...")
        t0 = time.perf_counter()
        model = dc.train_ecm(
            train_samples,
            lambdas.tolist(),
            n_classes=n_classes,
            T=T,
            alpha=args.alpha,
            use_clustering=args.use_clustering,
        )
        training_time = time.perf_counter() - t0
        logging.info(f"Training completed in {training_time:.2f}s")

        logging.info("Running inference...")
        if test_samples:
            _ = dc.simulate_stream_predict(model, test_samples[0])
        t0 = time.perf_counter()
        y_pred = []
        taus = []
        for s in test_samples:
            pred, tau = dc.simulate_stream_predict(model, s)
            y_pred.append(pred)
            taus.append(tau)
        inference_time = time.perf_counter() - t0
        y_pred = np.array(y_pred)
        taus = np.array(taus)

        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)
        earliness = float(np.mean(taus) / T)
        f_e = dc.harmonic_mean_f1_earliness(f1, earliness)
        inference_time_ms = (inference_time / len(test_samples)) * 1000 if test_samples else 0
        throughput_sps = len(test_samples) / inference_time if inference_time > 0 else 0

        logging.info(
            f"[Fold {fold_idx+1} Test] Acc: {accuracy:.4f} | F1: {f1:.4f} | "
            f"Earliness: {earliness:.4f} | F-E: {f_e:.4f} | "
            f"Inference: {inference_time_ms:.2f}ms | Throughput: {throughput_sps:.1f} sps"
        )

        fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "f1": f1,
            "earliness": earliness,
            "f_e": f_e,
            "inference_time_ms": inference_time_ms,
            "throughput_sps": throughput_sps,
            "training_time_sec": training_time,
        })
        all_predictions.extend(y_pred.tolist())
        all_true_labels.extend(test_labels.tolist())

    if not fold_results:
        logging.error("No fold results. Exiting.")
        return

    df = pd.DataFrame(fold_results)
    logging.info("\n" + "=" * 50)
    logging.info("K-Fold Summary")
    logging.info("=" * 50)
    for col in ["accuracy", "f1", "earliness", "f_e", "inference_time_ms", "throughput_sps", "training_time_sec"]:
        m, s = df[col].mean(), df[col].std()
        logging.info(f"{col.upper()}: {m:.4f} ± {s:.4f}")

    cm = confusion_matrix(all_true_labels, all_predictions)
    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    cm_norm = np.nan_to_num(cm_norm)

    df.to_csv(os.path.join(results_dir, "dc_per_fold_results.csv"), index=False)
    summary = df[["accuracy", "f1", "earliness", "f_e", "inference_time_ms", "throughput_sps", "training_time_sec"]].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(os.path.join(results_dir, "dc_kfold_summary.csv"))
    pd.DataFrame(cm).to_csv(os.path.join(results_dir, "dc_confusion_matrix.csv"), index=False, header=False)
    pd.DataFrame(cm_norm).to_csv(os.path.join(results_dir, "dc_confusion_matrix_normalized.csv"), index=False, header=False)

    logging.info(f"\nResults saved to {results_dir}/")
    logging.info("  - dc_per_fold_results.csv, dc_kfold_summary.csv, dc_confusion_matrix*.csv")


if __name__ == "__main__":
    main()
