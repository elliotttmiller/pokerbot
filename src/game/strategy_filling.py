"""
StrategyFilling: DeepStack-style strategy filling for missing/invalid strategy entries.
"""
import numpy as np

class StrategyFilling:
    def __init__(self):
        pass

    def fill_missing(self, strategy):
        # TODO: Fill missing/invalid entries with uniform probabilities
        strategy = np.array(strategy)
        mask = np.isnan(strategy)
        strategy[mask] = 1.0 / strategy.shape[0]
        return strategy
