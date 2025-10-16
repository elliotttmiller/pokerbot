#!/usr/bin/env python3
"""
Automated PokerBot Training Pipeline
Runs all steps: dependency check, data validation, training, validation, testing, and summary.
"""
import subprocess
import sys
import os


def run(cmd, desc):
    print(f"\n=== {desc} ===")
    env = os.environ.copy()
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    env['PYTHONPATH'] = src_path + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Step failed: {desc}")
        sys.exit(result.returncode)
    print(f"[OK] {desc} completed.")



# 2. Validate data
run('python scripts/validate_data.py', 'Validate training data')

# 3. Train PokerBot agent (standard mode, can change to production)
run('python scripts/train.py --agent-type pokerbot --mode standard --verbose --report', 'Train PokerBot agent')

# 4. Validate trained model
run('python scripts/validate_training.py --model models/versions/champion_best --hands 500', 'Validate trained model')

# 5. Run agent test suite
run('python examples/test_pokerbot.py', 'Test PokerBot agent')
run('python examples/validate_pokerbot.py', 'Validate PokerBot agent')

print("\n=== PokerBot Training Pipeline Complete! ===")
print("Check logs, models, and reports for results.")
