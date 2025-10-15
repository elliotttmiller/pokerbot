#!/usr/bin/env python3
"""CLI to validate training and equity datasets."""
import argparse
import json

from src.utils.data_validation import validate_deepstacked_samples, validate_equity_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default="data/deepstacked_training/train_samples")
    p.add_argument("--equity", default="data/equity_tables/preflop_equity.json")
    args = p.parse_args()

    res_samples = validate_deepstacked_samples(args.samples)
    res_equity = validate_equity_table(args.equity)

    print("Samples:")
    print(json.dumps(res_samples, indent=2))
    print("\nEquity:")
    print(json.dumps(res_equity, indent=2))


if __name__ == "__main__":
    main()
