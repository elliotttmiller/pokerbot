"""
StrategyFilling: DeepStack-style strategy filling for missing/invalid strategy entries.
"""
import numpy as np

class StrategyFilling:
    def __init__(self):
        pass

    def fill_missing(self, strategy):
        strategy = np.array(strategy, dtype=float)
        # Fill NaNs with uniform
        nan_mask = np.isnan(strategy)
        strategy[nan_mask] = 0.0
        # For each row, if sum is zero, fill with uniform
        for i in range(strategy.shape[0]):
            row = strategy[i]
            if np.sum(row) == 0.0:
                strategy[i] = 1.0 / strategy.shape[1]
            # Normalize row to sum to 1.0
            else:
                total = np.sum(row)
                if total > 0:
                    strategy[i] = row / total
        return strategy
