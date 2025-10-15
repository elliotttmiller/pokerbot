#!/usr/bin/env python3
"""Backward-compat champion training shim.

Forwards to the unified entrypoint: scripts/train.py (champion-only pipeline)
"""

import argparse
import sys
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='Champion training (forwarded to scripts/train.py)')
    parser.add_argument('--mode', type=str, choices=['smoketest', 'standard', 'production', 'full'], default='smoketest')
    parser.add_argument('--episodes', type=int, default=None)
    parser.add_argument('--cfr-iterations', type=int, default=None)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--verbose', action='store_true', default=False)
    args, unknown = parser.parse_known_args()

    # Resolve path to the unified train.py relative to this file
    repo_scripts_dir = os.path.dirname(os.path.abspath(__file__))
    train_py = os.path.join(repo_scripts_dir, 'train.py')

    cmd = [sys.executable, train_py, '--mode', args.mode, '--model-dir', args.model_dir]
    if args.episodes is not None:
        cmd += ['--episodes', str(args.episodes)]
    if args.cfr_iterations is not None:
        cmd += ['--cfr-iterations', str(args.cfr_iterations)]
    if args.verbose:
        cmd += ['--verbose']
    cmd += unknown

    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
