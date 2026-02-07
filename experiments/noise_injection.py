"""
Noise Injection Performance Experiment
- Top 3 baselines: LEC-GAN, CALIMERA, EARLIEST
- Datasets: OPPORTUNITY, CASAS
- Noise: x̃_t = x_t + ε, ε ~ N(0, σ²), σ = α * std(x)
- α ∈ {0.00, 0.10, 0.20, 0.30, 0.50}
- Continuous channels only (OPPORTUNITY: all 242; CASAS: all 37 for robustness)
"""
from __future__ import annotations

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.metrics import f1_score

from data_preprocessing.data_preprocess import pad_sequences


# =============================================================================
# Noise Injection
# =============================================================================

def compute_channel_std(train_data: List[np.ndarray], continuous_indices: List[int]) -> np.ndarray:
    """Compute per-channel std from training data (continuous channels only)."""
    flat = np.concatenate([seq[:, continuous_indices] for seq in train_data], axis=0)
    std_per_ch = np.std(flat, axis=0)
    std_per_ch[std_per_ch < 1e-12] = 1.0  # avoid division by zero
    return std_per_ch


def inject_noise(
    data: List[np.ndarray],
    alpha: float,
    channel_std_full: np.ndarray,
    continuous_indices: List[int],
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """
    Add Gaussian noise to continuous channels only.
    x̃_t = x_t + ε, ε ~ N(0, σ²), σ = α * std(x) per channel
    
    data: list of [T_i, C] arrays
    channel_std_full: std per channel for full dataset (shape [C])
    continuous_indices: which channels to apply noise
    """
    if alpha <= 0:
        return data
    
    noised = []
    for seq in data:
        seq = seq.copy()
        for j, ch_idx in enumerate(continuous_indices):
            sigma = alpha * channel_std_full[ch_idx]
            noise = rng.randn(seq.shape[0]) * sigma
            seq[:, ch_idx] = seq[:, ch_idx] + noise
        noised.append(seq)
    return noised


def get_continuous_channels(dataset: str, n_channels: int) -> List[int]:
    """
    OPPORTUNITY: all 242 channels are continuous (IMU)
    CASAS: 37 channels are binary - for robustness we apply to all (simulate sensor noise)
    """
    return list(range(n_channels))


# =============================================================================
# Metrics
# =============================================================================

def harmonic_mean_f1_earliness(f1: float, el: float, eps: float = 1e-12) -> float:
    t = 1.0 - el
    return float((2.0 * f1 * t) / (f1 + t + eps))


# =============================================================================
# Baseline Runners
# =============================================================================

def run_calimera_fit(
    train_data: List[np.ndarray],
    train_labels: np.ndarray,
    padding: str = "mean",
    delay_penalty: int = 1,
):
    from baselines.calimera import CALIMERA
    train_pad, _ = pad_sequences(train_data, padding_type=padding)
    X_train = train_pad.permute(0, 2, 1).numpy()  # (N, C, T)
    model = CALIMERA(delay_penalty=delay_penalty)
    model.fit(X_train, train_labels)
    return model


def run_calimera_test(model, test_data: List[np.ndarray], test_labels: np.ndarray, padding: str = "mean") -> Tuple[float, float, float]:
    test_pad, _ = pad_sequences(test_data, padding_type=padding)
    X_test = test_pad.permute(0, 2, 1).numpy()
    stop_ts, y_pred = model.test(X_test)
    y_pred = np.array(y_pred)
    T_max = X_test.shape[-1]
    earliness = np.mean([t / T_max for t in stop_ts])
    f1 = f1_score(test_labels, y_pred, average="macro", zero_division=0)
    hm = harmonic_mean_f1_earliness(f1, earliness)  # same as all baselines: HM = 2*F1*(1-EL)/(F1+(1-EL))
    return f1, earliness, hm


def run_earliest_fit(
    train_data: List[np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray = None,
    padding: str = "mean",
    device: torch.device = None,
    epochs: int = 30,
):
    from baselines.EARLIEST.model import EARLIEST
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pad, _ = pad_sequences(train_data, padding_type=padding)
    X_train = train_pad.permute(1, 0, 2).to(device)  # (T, N, C) = timesteps, batch, channels (match main_earliest)
    T, _, C = X_train.shape
    unique_labels = sorted(set(train_labels) | (set(test_labels) if test_labels is not None else set()))
    n_classes = len(unique_labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_train = np.array([label_map[y] for y in train_labels])
    
    class Args:
        rnn_cell = "LSTM"
        nhid = 64
        nlayers = 1
        lam = 0.1
        _epsilon = 0.1  # Controller exploration (match main_earliest)
    args = Args()
    model = EARLIEST(C, n_classes, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        perm = np.random.permutation(len(y_train))
        for i in range(0, len(perm), 32):
            idx = perm[i:i+32]
            x_b = X_train[:, idx, :]  # (T, batch, C)
            y_b = torch.tensor(y_train[idx], dtype=torch.long, device=device)
            logits, _ = model(x_b, epoch=epoch, test=False)
            loss = model.computeLoss(logits, y_b)  # RL + baseline + classification (match main_earliest)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, device, T, C, label_map


def run_earliest_test(model, device, T, C, label_map, test_data, test_labels, padding="mean") -> Tuple[float, float, float]:
    from baselines.EARLIEST.model import EARLIEST
    test_pad, _ = pad_sequences(test_data, padding_type=padding)
    X_test = test_pad.numpy() if isinstance(test_pad, torch.Tensor) else np.asarray(test_pad)
    # T=timesteps, C=channels (from fit). X_test shape (N, T_actual, C_actual).
    N_test, T_actual, C_actual = X_test.shape
    if T_actual != T or C_actual != C:
        X_new = np.zeros((N_test, T, C), dtype=np.float32)
        t_use, c_use = min(T_actual, T), min(C_actual, C)
        X_new[:, :t_use, :c_use] = X_test[:, :t_use, :c_use]
        if T_actual < T:
            X_new[:, t_use:, :c_use] = X_test[:, -1:, :c_use]
        if C_actual < C:
            X_new[:, :t_use, c_use:] = np.tile(X_test[:, :t_use, -1:], (1, 1, C - c_use))
            if T_actual < T:
                X_new[:, t_use:, c_use:] = np.tile(X_test[:, -1:, -1:], (1, T - t_use, C - c_use))
        X_test = X_new
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device).permute(1, 0, 2)  # (T, N, C)
    y_test = np.array([label_map[y] for y in test_labels])
    model.eval()
    model.Controller._epsilon = 0.0
    with torch.no_grad():
        logits, earl = model(X_test, epoch=0, test=True)
        y_pred = logits.argmax(dim=1).cpu().numpy()
        earliness = float(earl.item())  # match main_earliest
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    hm = harmonic_mean_f1_earliness(f1, earliness)
    return f1, earliness, hm


def _load_lecgan_mod():
    import importlib.util
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "baselines", "LEC-GAN", "lec-gan.py")
    spec = importlib.util.spec_from_file_location("lec_gan", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lec_gan"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_lecgan_fit(train_data, train_labels, test_labels=None, padding="mean", device=None, epochs=20):
    mod = _load_lecgan_mod()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_pad, _ = pad_sequences(train_data, padding_type=padding)
    X_train = train_pad.numpy() if isinstance(train_pad, torch.Tensor) else np.asarray(train_pad)
    if X_train.ndim == 2:
        X_train = X_train[:, :, np.newaxis]
    T, V = X_train.shape[1], X_train.shape[2]
    U = 4
    K = len(set(train_labels) | (set(test_labels) if test_labels is not None else set()))
    
    class LECGANDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.B = len(X)
            self.s = torch.randn(self.B, U)
            self.base_mask = torch.ones((self.B, T, V), dtype=torch.float32)
        def __len__(self): return self.B
        def __getitem__(self, i): return self.X[i], self.base_mask[i], self.s[i], self.y[i]
    
    cfg = mod.LECGANConfig(T=T, V=V, U=U, K=K, phi_rec=10.0, delta_adv=0.5, lr_g=1e-3, lr_d=1e-3)
    model = mod.LECGAN(cfg).to(device)
    opt_g = torch.optim.Adam(model.G.parameters(), lr=cfg.lr_g)
    opt_d = torch.optim.Adam(model.D.parameters(), lr=cfg.lr_d)
    mask_cfg = mod.EarlyMaskConfig(T=T, prefix_ratios=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], add_noise=True, noise_std=0.01)
    
    train_loader = torch.utils.data.DataLoader(LECGANDataset(X_train, train_labels), batch_size=32, shuffle=True)
    for _ in range(epochs):
        mod.train_one_epoch(model, train_loader, opt_g, opt_d, device, mask_cfg)
    return model, mod, device, T, V, U


def run_lecgan_test(model, mod, device, T, V, U, test_data, test_labels, padding="mean") -> Tuple[float, float, float]:
    test_pad, _ = pad_sequences(test_data, padding_type=padding)
    X_test = test_pad.numpy() if isinstance(test_pad, torch.Tensor) else np.asarray(test_pad)
    if X_test.ndim == 2:
        X_test = X_test[:, :, np.newaxis]
    # Ensure test has same T as training (pad or truncate)
    _, T_actual, _ = X_test.shape
    if T_actual != T:
        if T_actual < T:
            pad_width = ((0, 0), (0, T - T_actual), (0, 0))
            X_test = np.pad(X_test, pad_width, mode="edge")
        else:
            X_test = X_test[:, :T, :]
    
    class LECGANDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)
            self.B = len(X)
            self.s = torch.randn(self.B, U)
            self.base_mask = torch.ones((self.B, T, V), dtype=torch.float32)
        def __len__(self): return self.B
        def __getitem__(self, i): return self.X[i], self.base_mask[i], self.s[i], self.y[i]
    
    hcfg = mod.HaltingConfig(T=T, prefix_ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], use_maxprob=True, p_thresh=0.90,
                            use_stability=True, stability_runs=5, kl_thresh=0.02, imp_noise_std=0.01, min_ratio=0.1)
    test_loader = torch.utils.data.DataLoader(LECGANDataset(X_test, test_labels), batch_size=32, shuffle=False)
    metrics = mod.eval_early_metrics(model, test_loader, device, hcfg)
    return metrics["macro_f1"], metrics["earliness_el"], metrics["hm"]


# =============================================================================
# Main Experiment
# =============================================================================

ALPHA_LEVELS = [0.00, 0.10, 0.20, 0.30, 0.50]
TOP3_BASELINES = ["lecgan", "calimera", "earliest"]
DATASETS = ["opportunity", "casas"]
K_FOLD = 5


def run_experiment(
    datasets: List[str] = None,
    baselines: List[str] = None,
    k_fold: int = 5,
    max_folds: Optional[int] = None,
    seed: int = 42,
    quick_test: bool = False,
    epochs: Optional[int] = None,
    device: Optional[torch.device] = None,
    sample_ratios: Optional[dict] = None,
):
    datasets = datasets or DATASETS
    baselines = baselines or TOP3_BASELINES
    
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    
    results = []
    
    for dataset in datasets:
        fold_dir = f"fold_data/{dataset}"
        if not os.path.exists(fold_dir):
            print(f"[Skip] {dataset}: no fold data")
            continue
        
        n_folds = min(k_fold, max_folds or k_fold)
        for fold_idx in range(n_folds):
            fold_path = os.path.join(fold_dir, f"fold_{fold_idx}.pkl")
            if not os.path.exists(fold_path):
                break
            
            with open(fold_path, "rb") as f:
                fold = pickle.load(f)
            
            train_data = fold["train_data"]
            train_labels = np.array(fold["train_labels"])
            test_data = fold["test_data"]
            test_labels = np.array(fold["test_labels"])
            
            if quick_test:
                n = min(50, len(train_data))
                idx = rng.choice(len(train_data), n, replace=False)
                train_data = [train_data[i] for i in idx]
                train_labels = train_labels[idx]
                n = min(30, len(test_data))
                idx = rng.choice(len(test_data), n, replace=False)
                test_data = [test_data[i] for i in idx]
                test_labels = test_labels[idx]
            
            n_channels = train_data[0].shape[1]
            cont_channels = get_continuous_channels(dataset, n_channels)
            channel_std = compute_channel_std(train_data, cont_channels)
            channel_std_full = np.zeros(n_channels)
            for j, c in enumerate(cont_channels):
                channel_std_full[c] = channel_std[j]
            
            for baseline in baselines:
                n_epochs = epochs if epochs is not None else (10 if quick_test else 30)
                print(f"  [Fit] {dataset} fold{fold_idx} {baseline} (epochs={n_epochs})...", flush=True)
                try:
                    if baseline == "calimera":
                        model = run_calimera_fit(train_data, train_labels, delay_penalty=1)
                    elif baseline == "earliest":
                        model, device, T, C, label_map = run_earliest_fit(
                            train_data, train_labels, test_labels=test_labels, epochs=n_epochs, device=device
                        )
                    elif baseline == "lecgan":
                        model, mod, device, T, V, U = run_lecgan_fit(
                            train_data, train_labels, test_labels=test_labels, epochs=n_epochs, device=device
                        )
                except Exception as e:
                    print(f"  [TRAIN FAIL] {dataset} fold{fold_idx} {baseline}: {e}")
                    continue
                
                for alpha in ALPHA_LEVELS:
                    test_noised = inject_noise(test_data, alpha, channel_std_full, cont_channels, rng)
                    try:
                        if baseline == "calimera":
                            f1, el, hm = run_calimera_test(model, test_noised, test_labels)
                        elif baseline == "earliest":
                            f1, el, hm = run_earliest_test(
                                model, device, T, C, label_map, test_noised, test_labels
                            )
                        elif baseline == "lecgan":
                            f1, el, hm = run_lecgan_test(
                                model, mod, device, T, V, U, test_noised, test_labels
                            )
                        results.append({
                            "dataset": dataset,
                            "baseline": baseline,
                            "fold": fold_idx,
                            "alpha": alpha,
                            "f1": f1,
                            "earliness": el,
                            "hm": hm,
                        })
                        print(f"  {dataset} fold{fold_idx} {baseline} α={alpha:.2f} -> HM={hm:.4f}")
                    except Exception as e:
                        import traceback
                        print(f"  [FAIL] {dataset} fold{fold_idx} {baseline} α={alpha:.2f}: {e}")
                        traceback.print_exc()
    
    return pd.DataFrame(results)


def aggregate_and_report(df: pd.DataFrame, output_dir: str = "results/noise_injection", output_suffix: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    
    agg = df.groupby(["dataset", "baseline", "alpha"]).agg({
        "hm": ["mean", "std"],
        "f1": "mean",
        "earliness": "mean",
    }).reset_index()
    
    agg.columns = ["dataset", "baseline", "alpha", "hm_mean", "hm_std", "f1_mean", "el_mean"]
    
    hm_baseline = df[df["alpha"] == 0].groupby(["dataset", "baseline"])["hm"].mean().to_dict()
    
    def get_baseline_hm(ds, bl):
        return hm_baseline.get((ds, bl), 0.0)
    
    agg["hm_drop"] = agg.apply(
        lambda r: get_baseline_hm(r["dataset"], r["baseline"]) - r["hm_mean"],
        axis=1
    )
    
    out_csv = os.path.join(output_dir, f"noise_results{output_suffix}.csv")
    agg.to_csv(out_csv, index=False)
    
    # Report
    lines = [
        "# Noise Injection Performance Report",
        "",
        "## HM (mean ± std) by α",
        "",
    ]
    
    for dataset in agg["dataset"].unique():
        lines.append(f"### {dataset.upper()}")
        lines.append("")
        for baseline in ["lecgan", "calimera", "earliest"]:
            sub = agg[(agg["dataset"] == dataset) & (agg["baseline"] == baseline)]
            if sub.empty:
                continue
            lines.append(f"**{baseline.upper()}**")
            for _, r in sub.iterrows():
                lines.append(f"- α={r['alpha']:.2f}: HM={r['hm_mean']:.4f}±{r['hm_std']:.4f}, ΔHM={r['hm_drop']:.4f}, F1={r['f1_mean']:.4f}, EL={r['el_mean']:.4f}")
            lines.append("")
        lines.append("")
    
    report_path = os.path.join(output_dir, "noise_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report: {report_path}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax_idx, dataset in enumerate(agg["dataset"].unique()):
        ax = axes[ax_idx]
        for baseline in ["lecgan", "calimera", "earliest"]:
            sub = agg[(agg["dataset"] == dataset) & (agg["baseline"] == baseline)].sort_values("alpha")
            if sub.empty:
                continue
            ax.errorbar(
                sub["alpha"],
                sub["hm_mean"],
                yerr=sub["hm_std"],
                marker="o",
                label=baseline.upper(),
                capsize=3,
            )
        ax.set_xlabel("α (noise strength)")
        ax.set_ylabel("HM (Harmonic Mean)")
        ax.set_title(f"{dataset.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ALPHA_LEVELS)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "noise_hm_plot.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot: {plot_path}")
    
    # ΔHM plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    for ax_idx, dataset in enumerate(agg["dataset"].unique()):
        ax = axes[ax_idx]
        for baseline in ["lecgan", "calimera", "earliest"]:
            sub = agg[(agg["dataset"] == dataset) & (agg["baseline"] == baseline)].sort_values("alpha")
            if sub.empty:
                continue
            ax.plot(sub["alpha"], sub["hm_drop"], marker="o", label=baseline.upper())
        ax.set_xlabel("α (noise strength)")
        ax.set_ylabel("ΔHM (HM drop)")
        ax.set_title(f"{dataset.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(ALPHA_LEVELS)
    
    plt.tight_layout()
    plot_path2 = os.path.join(output_dir, "noise_dhm_plot.png")
    plt.savefig(plot_path2, dpi=150)
    plt.close()
    print(f"ΔHM Plot: {plot_path2}")


def run_dry_run(output_dir: str = "results/noise_injection"):
    """Generate report/plot from fake data for pipeline verification."""
    np.random.seed(42)
    rows = []
    for dataset in DATASETS:
        for baseline in TOP3_BASELINES:
            for fold in range(3):
                hm_base = 0.7 + np.random.rand() * 0.2
                for alpha in ALPHA_LEVELS:
                    hm = hm_base * (1 - alpha * 0.5) + np.random.randn() * 0.05
                    hm = max(0.01, min(1.0, hm))
                    f1 = hm * 0.9 + np.random.randn() * 0.05
                    el = 0.1 + alpha * 0.2 + np.random.randn() * 0.02
                    rows.append({
                        "dataset": dataset, "baseline": baseline, "fold": fold,
                        "alpha": alpha, "f1": max(0, f1), "earliness": max(0, min(1, el)), "hm": hm
                    })
    df = pd.DataFrame(rows)
    aggregate_and_report(df, output_dir)
    print("Dry run complete. Check outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--baselines", nargs="+", default=TOP3_BASELINES)
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument("--quick_test", action="store_true", help="Small subset for quick run")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs (default: 10 if quick_test else 30)")
    parser.add_argument("--max_folds", type=int, default=None, help="Limit folds (default: all)")
    parser.add_argument("--output_dir", type=str, default="results/noise_injection")
    parser.add_argument("--output_suffix", type=str, default="", help="e.g. _casas for partial run")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    parser.add_argument("--dry_run", action="store_true", help="Fake data to test report/plot pipeline")
    parser.add_argument("--sample", nargs=2, action="append", metavar=("DATASET", "RATIO"),
                        help="Sample ratio per dataset e.g. --sample openpack 0.3")
    parser.add_argument("--merge", nargs="+", metavar="CSV", help="Merge partial CSVs and regenerate report")
    args = parser.parse_args()
    
    if args.merge:
        dfs = [pd.read_csv(f) for f in args.merge]
        df = pd.concat(dfs, ignore_index=True)
        aggregate_and_report(df, args.output_dir)
        print("Merged and report generated.")
    elif args.dry_run:
        run_dry_run(args.output_dir)
    else:
        dev = torch.device("cpu") if args.cpu else None
        sample_ratios = {d: float(r) for d, r in (args.sample or [])}
        df = run_experiment(
            datasets=args.datasets,
            baselines=args.baselines,
            k_fold=args.k_fold,
            max_folds=args.max_folds,
            quick_test=args.quick_test,
            epochs=args.epochs,
            device=dev,
            sample_ratios=sample_ratios or None,
        )
        
        if not df.empty:
            if args.output_suffix:
                raw_path = os.path.join(args.output_dir, f"noise_raw{args.output_suffix}.csv")
                os.makedirs(args.output_dir, exist_ok=True)
                df.to_csv(raw_path, index=False)
                print(f"Saved raw results: {raw_path}")
            aggregate_and_report(df, args.output_dir, output_suffix=args.output_suffix)
            print("\nDone.")
        else:
            print("No results.")
