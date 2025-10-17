#!/usr/bin/env python3
"""
Derive 169-length bucket sampling weights from validator per-bucket correlations.

Usage:
    python scripts/derive_bucket_weights.py --corr-json models/reports/per_bucket_corrs.json --out C:\path\to\bucket_weights.json --boost 3.0 --bottom 40

Logic:
  - Start from ones.
  - Identify the bottom-K buckets by correlation (default K=40) and set their weights to `boost`.
  - Optionally apply a smooth mapping: weight = exp(-corr * scale), but default keeps it simple.
"""
import os
import json
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corr-json', type=str, required=True, help='Path to per_bucket_corrs.json')
    ap.add_argument('--out', type=str, required=True, help='Output JSON path for 169-length weights')
    ap.add_argument('--boost', type=float, default=3.0, help='Weight value to assign to worst buckets')
    ap.add_argument('--bottom', type=int, default=40, help='How many worst buckets to boost')
    args = ap.parse_args()

    with open(args.corr_json, 'r') as f:
        data = json.load(f)
    corrs = np.asarray(data.get('bucket_corrs', []), dtype=np.float64)
    nb = int(data.get('num_buckets', len(corrs)))
    if len(corrs) != nb:
        raise ValueError(f"Mismatch corr length {len(corrs)} vs num_buckets {nb}")
    if nb != 169:
        print(f"[WARN] Expected 169 buckets, got {nb} — proceeding anyway")

    weights = np.ones(nb, dtype=np.float64)
    # Worst-K by correlation (ascending); handle NaNs by treating as very low
    safe_corrs = np.where(np.isnan(corrs), -1.0, corrs)
    worst_idx = np.argsort(safe_corrs)[:max(1, min(args.bottom, nb))]
    weights[worst_idx] = float(args.boost)

    with open(args.out, 'w') as f:
        json.dump(list(map(float, weights.tolist())), f)
    print(f"Wrote weights to {args.out}; boosted {len(worst_idx)} buckets to {args.boost}")


if __name__ == '__main__':
    main()
