#!/usr/bin/env python3
"""CLI to validate training and equity datasets."""
import argparse
import json
import os
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
# Fallback: always add src path directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.utils.data_validation import validate_deepstacked_samples, validate_equity_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", default=r"C:\Users\AMD\pokerbot\src\train_samples")
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
