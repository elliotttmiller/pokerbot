"""
Mock neural network terminal for DeepStack pipeline testing.
Ported from DeepStack Lua mock_nn_terminal.lua.
"""
import numpy as np

class MockNnTerminal:
    def __init__(self, bucket_count):
        self.bucket_count = bucket_count
        self.equity_matrix = np.zeros((bucket_count, bucket_count))
        # Fill equity_matrix with dummy values for demonstration
        for i in range(bucket_count):
            for j in range(bucket_count):
                self.equity_matrix[i, j] = 1.0 if i == j else 0.5

    def get_value(self, inputs):
        batch_size = inputs.shape[0]
        outputs = np.zeros((batch_size, self.bucket_count * 2))
        for b in range(batch_size):
            for player in range(2):
                idx = player * self.bucket_count
                outputs[b, idx:idx+self.bucket_count] = np.dot(inputs[b, idx:idx+self.bucket_count], self.equity_matrix)
        return outputs

def mock_terminal_nn(bucket_count):
    """Create a mock terminal NN object."""
    return MockNnTerminal(bucket_count)
