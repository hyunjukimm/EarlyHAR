"""
Collect F1, earliness, and harmonic mean from baseline results across datasets.
Reads results/{dataset}/{baseline}_kfold_summary.csv and produces baseline_summary_all.md.
"""
import argparse
import os
import pandas as pd


BASELINES = ["calimera", "dc", "earliest", "stopandhop", "attn", "teaser", "lecgan"]
METRICS = ["f1", "earliness", "f_e"]


def setup_args():
    parser = argparse.ArgumentParser(description="Collect baseline metrics")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/baseline_summary_all.md")
    parser.add_argument("--datasets", type=str, nargs="*", default=None, help="Filter datasets")
    return parser.parse_args()


def collect_metrics(results_dir, datasets=None):
    data = []
    if datasets is None:
        datasets = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for dataset in sorted(datasets):
        dataset_dir = os.path.join(results_dir, dataset)
        if not os.path.isdir(dataset_dir):
            continue
        for baseline in BASELINES:
            path = os.path.join(dataset_dir, f"{baseline}_kfold_summary.csv")
            if not os.path.exists(path):
                continue
            try:
                df = pd.read_csv(path, index_col=0)
                for metric in METRICS:
                    if metric not in df.index:
                        continue
                    row = df.loc[metric]
                    mean_val = row.get("mean", row.iloc[0]) if hasattr(row, "get") else row[0]
                    std_val = row.get("std", row.iloc[1]) if hasattr(row, "get") else row[1]
                    data.append({
                        "dataset": dataset,
                        "baseline": baseline,
                        "metric": metric,
                        "mean": mean_val,
                        "std": std_val,
                    })
            except Exception as e:
                print(f"Skip {path}: {e}")
    return pd.DataFrame(data)


def to_markdown(df, output_path):
    if df.empty:
        with open(output_path, "w") as f:
            f.write("# Baseline Summary\n\nNo results found.\n")
        return
    lines = ["# Baseline Summary (F1, Earliness, F-E)\n"]
    for dataset in df["dataset"].unique():
        lines.append(f"\n## {dataset}\n")
        sub = df[df["dataset"] == dataset].copy()
        for baseline in sub["baseline"].unique():
            bsub = sub[sub["baseline"] == baseline]
            parts = []
            for _, r in bsub.iterrows():
                parts.append(f"{r['metric']}: {r['mean']:.4f}±{r['std']:.4f}")
            lines.append(f"- **{baseline}**: {', '.join(parts)}\n")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")


def main():
    args = setup_args()
    df = collect_metrics(args.results_dir, args.datasets)
    to_markdown(df, args.output)


if __name__ == "__main__":
    main()
