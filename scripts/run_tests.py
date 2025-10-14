#!/usr/bin/env python3
"""Run all tests for the poker bot system."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run tests
from examples import test, test_champion

def main():
    """Run all test suites."""
    print("="*70)
    print("RUNNING ALL POKER BOT TESTS")
    print("="*70)
    print()
    
    # Run base tests
    print("Running base tests...")
    test_result = os.system(f"cd {os.path.dirname(__file__)} && python3 examples/test.py")
    
    print("\n" + "="*70)
    
    # Run champion tests
    print("Running champion agent tests...")
    champion_result = os.system(f"cd {os.path.dirname(__file__)} && python3 examples/test_champion.py")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    
    sys.exit(0 if (test_result == 0 and champion_result == 0) else 1)

if __name__ == '__main__':
    main()
