#!/usr/bin/env python3
"""Collect f1, earliness, HM (f_e) mean±std from results/*/baseline_kfold_summary.csv"""
import pandas as pd
from pathlib import Path

results_dir = Path("results")
datasets = ["aras", "casas", "doore", "openpack", "opportunity"]
baselines = ["earliest", "stopandhop", "calimera", "teaser", "attn", "dc", "lecgan"]


def get_metric(ds, bl, name):
    p = results_dir / ds / f"{bl}_kfold_summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, index_col=0)
    if name not in df.index:
        return None
    m, s = df.loc[name, "mean"], df.loc[name, "std"]
    if pd.isna(s):
        s = 0.0
    return f"{float(m):.4f}±{float(s):.4f}"


rows = []
for ds in datasets:
    for bl in baselines:
        f1 = get_metric(ds, bl, "f1")
        if f1 is None:
            continue
        ear = get_metric(ds, bl, "earliness")
        hm = get_metric(ds, bl, "f_e")
        rows.append({"dataset": ds, "baseline": bl, "f1": f1, "earliness": ear, "HM": hm})

for ds in datasets:
    sub = [r for r in rows if r["dataset"] == ds]
    if not sub:
        continue
    print(f"=== {ds} ===")
    for r in sub:
        b, f, e, h = r["baseline"], r["f1"], r["earliness"], r["HM"]
        print(f"  {b:12} | f1={f} | earliness={e} | HM={h}")
    print()

# Also write CSV
out = Path("results/summary_f1_earliness_hm.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Saved: {out}")
