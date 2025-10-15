"""
MaskedHuberLoss: DeepStack-style masked Huber loss for neural network training.
"""
import numpy as np

def masked_huber_loss(y_true, y_pred, mask, delta=1.0):
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss * mask
    return np.sum(loss) / np.sum(mask)
