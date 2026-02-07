"""
TEASER baseline - sktime TEASER for early time series classification.
Uses fold data, pad_sequences. Falls back to 1-NN if sktime unavailable.
"""
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baselines.TEASER import create_teaser_model, fit_teaser, predict_teaser, fallback_1nn_predict
from data_preprocessing.data_preprocess import pad_sequences


def setup_args():
    parser = argparse.ArgumentParser(description="TEASER baseline")
    parser.add_argument("--dataset", type=str, default="casas")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--padding", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def harmonic_mean_f1_earliness(f1, earliness):
    return 2 * ((1 - earliness) * f1) / ((1 - earliness) + f1 + 1e-8)


def main():
    args = setup_args()
    np.random.seed(args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = os.path.join("logs", f"teaser_{args.dataset}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    results_dir = os.path.join("results", args.dataset)
    os.makedirs(results_dir, exist_ok=True)
    fold_results = []

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

        if args.sample_ratio is not None:
            rng = np.random.RandomState(args.seed)
            n_train = max(30, int(len(train_data) * args.sample_ratio))
            n_test = max(20, int(len(test_data) * args.sample_ratio))
            if len(train_data) > n_train:
                idx = rng.choice(len(train_data), n_train, replace=False)
                train_data = [train_data[i] for i in idx]
                train_labels = train_labels[idx]
            if len(test_data) > n_test:
                idx = rng.choice(len(test_data), n_test, replace=False)
                test_data = [test_data[i] for i in idx]
                test_labels = test_labels[idx]

        if args.max_train is not None and len(train_data) > args.max_train:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(train_data), args.max_train, replace=False)
            train_data = [train_data[i] for i in idx]
            train_labels = train_labels[idx]
        if args.max_test is not None and len(test_data) > args.max_test:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(len(test_data), args.max_test, replace=False)
            test_data = [test_data[i] for i in idx]
            test_labels = test_labels[idx]

        train_padded, _ = pad_sequences(train_data, padding_type=args.padding)
        test_padded, _ = pad_sequences(test_data, padding_type=args.padding)

        X_train = np.asarray(train_padded) if not hasattr(train_padded, "numpy") else train_padded.numpy()
        X_test = np.asarray(test_padded) if not hasattr(test_padded, "numpy") else test_padded.numpy()

        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]

        T = X_train.shape[1]
        model = create_teaser_model()
        model = fit_teaser(model, X_train, train_labels)

        if model is not None:
            y_pred, earliness = predict_teaser(model, X_test)
        else:
            logging.info("TEASER unavailable, using 1-NN fallback")
            y_pred, earliness = fallback_1nn_predict(X_train, train_labels, X_test)

        if y_pred is None:
            y_pred = np.zeros(len(test_labels), dtype=int)
            earliness = 0.0

        accuracy = accuracy_score(test_labels, y_pred)
        f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)
        f_e = harmonic_mean_f1_earliness(f1, earliness)

        logging.info(
            f"[Fold {fold_idx+1}] Acc: {accuracy:.4f} | F1: {f1:.4f} | "
            f"Earliness: {earliness:.4f} | F-E: {f_e:.4f}"
        )

        fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "f1": f1,
            "earliness": earliness,
            "f_e": f_e,
        })

    if not fold_results:
        logging.error("No fold results. Exiting.")
        return

    df = pd.DataFrame(fold_results)
    for col in ["accuracy", "f1", "earliness", "f_e"]:
        m, s = df[col].mean(), df[col].std()
        logging.info(f"{col.upper()}: {m:.4f} ± {s:.4f}")

    summary = df[["accuracy", "f1", "earliness", "f_e"]].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(os.path.join(results_dir, "teaser_kfold_summary.csv"))
    df.to_csv(os.path.join(results_dir, "teaser_per_fold_results.csv"), index=False)
    logging.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
