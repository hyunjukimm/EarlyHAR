"""
Compute across-datasets mean for each baseline.
Reads from results/summary_f1_earliness_hm.csv and computes mean metrics for each baseline.
"""
import pandas as pd
import numpy as np


def parse_mean_std(s):
    """Parse '0.1234±0.0567' into (mean, std)"""
    if '±' not in str(s):
        return float(s), 0.0
    parts = str(s).split('±')
    return float(parts[0]), float(parts[1])


def main():
    # Read summary CSV
    df = pd.read_csv('results/summary_f1_earliness_hm.csv')
    
    print("=" * 80)
    print("ACROSS-DATASETS MEAN FOR EACH BASELINE")
    print("=" * 80)
    print()
    
    baselines = df['baseline'].unique()
    datasets = df['dataset'].unique()
    
    print(f"Datasets: {', '.join(datasets)} (n={len(datasets)})")
    print()
    
    # Prepare results table
    results = []
    
    for baseline in sorted(baselines):
        baseline_data = df[df['baseline'] == baseline]
        
        # Parse mean values for each metric
        f1_means = []
        earliness_means = []
        hm_means = []
        
        for _, row in baseline_data.iterrows():
            f1_mean, _ = parse_mean_std(row['f1'])
            el_mean, _ = parse_mean_std(row['earliness'])
            hm_mean, _ = parse_mean_std(row['HM'])
            
            f1_means.append(f1_mean)
            earliness_means.append(el_mean)
            hm_means.append(hm_mean)
        
        # Compute across-datasets statistics
        f1_avg = np.mean(f1_means)
        f1_std = np.std(f1_means, ddof=1) if len(f1_means) > 1 else 0.0
        
        el_avg = np.mean(earliness_means)
        el_std = np.std(earliness_means, ddof=1) if len(earliness_means) > 1 else 0.0
        
        hm_avg = np.mean(hm_means)
        hm_std = np.std(hm_means, ddof=1) if len(hm_means) > 1 else 0.0
        
        results.append({
            'baseline': baseline.upper(),
            'f1': f"{f1_avg:.4f}±{f1_std:.4f}",
            'earliness': f"{el_avg:.4f}±{el_std:.4f}",
            'HM': f"{hm_avg:.4f}±{hm_std:.4f}",
            'f1_mean': f1_avg,  # for sorting
            'hm_mean': hm_avg,
        })
    
    # Create DataFrame and sort by HM
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('hm_mean', ascending=False)
    
    # Print table
    print("| Baseline    | F1 (across)    | Earliness (across) | HM (across)    |")
    print("|-------------|----------------|-------------------|----------------|")
    for _, row in results_df.iterrows():
        print(f"| {row['baseline']:11s} | {row['f1']:14s} | {row['earliness']:17s} | {row['HM']:14s} |")
    
    print()
    print("=" * 80)
    print("NOTES:")
    print("- F1: Macro-F1 score (higher is better)")
    print("- Earliness: Prediction time ratio (lower is better, 0=early, 1=late)")
    print("- HM: Harmonic Mean of F1 and (1-Earliness) (higher is better)")
    print("- Values are mean±std across {} datasets".format(len(datasets)))
    print("=" * 80)
    
    # Save to file
    output_df = results_df[['baseline', 'f1', 'earliness', 'HM']].copy()
    output_df.to_csv('results/baseline_means_across_datasets.csv', index=False)
    print("\nSaved to: results/baseline_means_across_datasets.csv")
    
    # Print ranking by HM
    print("\n" + "=" * 80)
    print("RANKING BY HARMONIC MEAN (F-E):")
    print("=" * 80)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i}. {row['baseline']:11s} - HM: {row['HM']}")
    print()


if __name__ == "__main__":
    main()
