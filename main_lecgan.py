"""
LEC-GAN baseline - Full GAN implementation with halting policy.
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
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

from data_preprocessing.data_preprocess import pad_sequences


def _load_lecgan():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "baselines", "LEC-GAN", "lec-gan.py")
    spec = importlib.util.spec_from_file_location("lec_gan", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lec_gan"] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_args():
    parser = argparse.ArgumentParser(description="LEC-GAN baseline (Full GAN)")
    parser.add_argument("--dataset", type=str, default="casas")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--padding", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--nhid", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr_g", type=float, default=1e-3)
    parser.add_argument("--lr_d", type=float, default=1e-3)
    parser.add_argument("--phi_rec", type=float, default=10.0)
    parser.add_argument("--delta_adv", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_ratio", type=float, default=None)
    parser.add_argument("--max_train", type=int, default=None)
    parser.add_argument("--max_test", type=int, default=None)
    parser.add_argument("--p_thresh", type=float, default=0.90)
    parser.add_argument("--stability_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class LECGANDataset(Dataset):
    """Dataset for LEC-GAN that returns (x, base_mask, s, y)"""
    def __init__(self, X, y, static_dim=4):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.B, self.T, self.V = self.X.shape
        # Create dummy static features (you can replace with real static features)
        self.s = torch.randn(self.B, static_dim)
        # Base mask: assume all observed (1.0)
        self.base_mask = torch.ones((self.B, self.T, self.V), dtype=torch.float32)
    
    def __len__(self):
        return self.B
    
    def __getitem__(self, idx):
        return self.X[idx], self.base_mask[idx], self.s[idx], self.y[idx]


def main():
    args = setup_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    lecgan_module = _load_lecgan()
    lecgan_module.set_seed(args.seed)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    log_file = os.path.join("logs", f"lecgan_{args.dataset}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        X_train = train_padded.numpy() if isinstance(train_padded, torch.Tensor) else np.asarray(train_padded)
        X_test = test_padded.numpy() if isinstance(test_padded, torch.Tensor) else np.asarray(test_padded)

        if X_train.ndim == 2:
            X_train = X_train[:, :, np.newaxis]
            X_test = X_test[:, :, np.newaxis]

        T, V = X_train.shape[1], X_train.shape[2]
        U = 4  # Static feature dimension
        K = len(set(train_labels) | set(test_labels))

        # Create datasets and loaders
        train_dataset = LECGANDataset(X_train, train_labels, static_dim=U)
        test_dataset = LECGANDataset(X_test, test_labels, static_dim=U)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Create LEC-GAN model
        cfg = lecgan_module.LECGANConfig(
            T=T, V=V, U=U, K=K,
            phi_rec=args.phi_rec,
            delta_adv=args.delta_adv,
            lr_g=args.lr_g,
            lr_d=args.lr_d
        )
        model = lecgan_module.LECGAN(cfg).to(device)

        # Training configuration
        mask_cfg = lecgan_module.EarlyMaskConfig(
            T=T,
            prefix_ratios=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
            add_noise=True,
            noise_std=0.01,
        )

        # Halting configuration for evaluation
        hcfg = lecgan_module.HaltingConfig(
            T=T,
            prefix_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            use_maxprob=True,
            p_thresh=args.p_thresh,
            use_entropy=False,
            use_stability=True,
            stability_runs=args.stability_runs,
            kl_thresh=0.02,
            imp_noise_std=0.01,
            min_ratio=0.1,
        )

        # Train model
        logging.info("Training LEC-GAN...")
        start_train = time.time()
        opt_g = torch.optim.Adam(model.G.parameters(), lr=cfg.lr_g)
        opt_d = torch.optim.Adam(model.D.parameters(), lr=cfg.lr_d)

        for epoch in range(1, args.epochs + 1):
            tr = lecgan_module.train_one_epoch(model, train_loader, opt_g, opt_d, device, mask_cfg)
            if epoch % 5 == 0 or epoch == 1:
                logging.info(
                    f"Epoch {epoch:03d} | "
                    f"g_loss {tr['g_loss']:.4f} d_loss {tr['d_loss']:.4f} "
                    f"g_rec {tr['g_rec']:.4f} g_fm {tr['g_fm']:.4f}"
                )

        train_time = time.time() - start_train

        # Evaluate with halting policy
        logging.info("Evaluating with halting policy...")
        start_inf = time.time()
        eval_result = lecgan_module.eval_early_metrics(model, test_loader, device, hcfg)
        inference_time = time.time() - start_inf

        f1 = eval_result["macro_f1"]
        earliness = eval_result["earliness_el"]
        hm = eval_result["hm"]

        # Also compute accuracy for comparison
        model.eval()
        all_pred = []
        all_y = []
        with torch.no_grad():
            for batch in test_loader:
                x, base_mask, s, y = batch
                x = x.to(device)
                base_mask = base_mask.to(device)
                s = s.to(device)
                pred, _ = lecgan_module.early_predict_with_halting(model, x, base_mask, s, hcfg)
                all_pred.append(pred.cpu())
                all_y.append(y)
        
        y_pred = torch.cat(all_pred).numpy()
        y_true = torch.cat(all_y).numpy()
        accuracy = accuracy_score(y_true, y_pred)

        inference_time_ms = (inference_time / len(test_labels)) * 1000
        throughput_sps = len(test_labels) / inference_time if inference_time > 0 else 0

        logging.info(
            f"[Fold {fold_idx+1}] Acc: {accuracy:.4f} | F1: {f1:.4f} | "
            f"Earliness: {earliness:.4f} | HM: {hm:.4f}"
        )
        logging.info(
            f"Training time: {train_time:.2f}s | "
            f"Inference: {inference_time_ms:.2f}ms/sample | "
            f"Throughput: {throughput_sps:.2f} samples/s"
        )

        fold_results.append({
            "fold": fold_idx,
            "accuracy": accuracy,
            "f1": f1,
            "earliness": earliness,
            "f_e": hm,
            "training_time_sec": train_time,
            "inference_time_ms": inference_time_ms,
            "throughput_sps": throughput_sps,
        })

    if not fold_results:
        logging.error("No fold results. Exiting.")
        return

    df = pd.DataFrame(fold_results)
    for col in ["accuracy", "f1", "earliness", "f_e", "training_time_sec", "inference_time_ms", "throughput_sps"]:
        if col in df.columns:
            m, s = df[col].mean(), df[col].std()
            logging.info(f"{col.upper()}: {m:.4f} ± {s:.4f}")

    summary = df[["accuracy", "f1", "earliness", "f_e", "training_time_sec", "inference_time_ms", "throughput_sps"]].agg(["mean", "std"]).T
    summary.index.name = "metric"
    summary.to_csv(os.path.join(results_dir, "lecgan_kfold_summary.csv"))
    df.to_csv(os.path.join(results_dir, "lecgan_per_fold_results.csv"), index=False)
    logging.info(f"Results saved to {results_dir}/")


if __name__ == "__main__":
    main()
