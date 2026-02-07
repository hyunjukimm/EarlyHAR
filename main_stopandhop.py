"""
StopAndHop baseline - halting policy for early classification.
Uses fold data, pad_sequences, trains StopAndHop, evaluates with earliness/F1/HM.
"""
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from baselines.StopAndHop.model import StopAndHop
from data_preprocessing.data_preprocess import pad_sequences


def setup_args():
    parser = argparse.ArgumentParser(description="StopAndHop baseline")
    parser.add_argument("--dataset", type=str, default="casas")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--padding", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lam", type=float, default=0.1)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def harmonic_mean_f1_earliness(f1, earliness):
    return 2 * ((1 - earliness) * f1) / ((1 - earliness) + f1 + 1e-8)


def main():
    args = setup_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = os.path.join("logs", f"stopandhop_{args.dataset}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        X_train = train_padded.numpy() if isinstance(train_padded, torch.Tensor) else np.asarray(train_padded)
        X_test = test_padded.numpy() if isinstance(test_padded, torch.Tensor) else np.asarray(test_padded)

        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]

        T, C = X_train.shape[1], X_train.shape[2]
        n_classes = len(set(train_labels) | set(test_labels))

        config = {
            "n_epochs": args.n_epochs,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "nhid": args.nhid,
        }
        data_setting = {
            "N_FEATURES": C,
            "N_CLASSES": n_classes,
            "nsteps": T,
        }

        model = StopAndHop(config, data_setting, lam=args.lam).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.n_epochs):
            model.train()
            perm = np.random.permutation(len(X_train))
            total_loss = 0.0
            n_batches = 0
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i : i + args.batch_size]
                X_b = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
                y_b = torch.tensor(train_labels[idx], dtype=torch.long).to(device)
                logits, _, _ = model(X_b, test=False, epoch=epoch)
                loss = model.computeLoss(logits, y_b)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{args.n_epochs} loss={total_loss/max(n_batches,1):.4f}")

        model.eval()
        y_pred = []
        earliness_list = []
        with torch.no_grad():
            for i in range(0, len(X_test), args.batch_size):
                X_b = torch.tensor(X_test[i : i + args.batch_size], dtype=torch.float32).to(device)
                logits, _, halt_points = model(X_b, test=True, epoch=0)
                pred = logits.argmax(dim=1).cpu().numpy()
                y_pred.extend(pred.tolist())
                taus = halt_points.cpu().numpy()
                earliness_list.extend((taus / T).tolist())
        y_pred = np.array(y_pred)
        earliness = float(np.mean(earliness_list)) if earliness_list else 0.0

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
        all_predictions.extend(y_pred.tolist())
        all_true_labels.extend(test_labels.tolist())

    if not fold_results:
        logging.error("No fold results. Exiting.")
        return

    df = pd.DataFrame(fold_results)
    for col in ["accuracy", "f1", "earliness", "f_e"]:
        m, s = df[col].mean(), df[col].std()
        logging.info(f"{col.upper()}: {m:.4f} ± {s:.4f}")

    summary = df[["accuracy", "f1", "earliness", "f_e"]].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(os.path.join(results_dir, "stopandhop_kfold_summary.csv"))
    df.to_csv(os.path.join(results_dir, "stopandhop_per_fold_results.csv"), index=False)
    logging.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
