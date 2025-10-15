"""
Range generator for DeepStack data generation.
Ported from DeepStack Lua range_generator.lua.
"""
import numpy as np

def generate_range(batch_size, range_size):
    """
    Samples a batch of random probability vectors for player ranges.
    Args:
        batch_size: number of ranges to sample
        range_size: size of each range vector
    Returns:
        np.ndarray of shape (batch_size, range_size) with each row summing to 1
    """
    ranges = np.random.dirichlet(np.ones(range_size), batch_size)
    return ranges

# Example usage:
# ranges = generate_range(32, 169)
