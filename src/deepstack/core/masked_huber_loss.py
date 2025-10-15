"""
MaskedHuberLoss: DeepStack-style masked Huber loss for neural network training.
"""
import numpy as np
import torch
import torch.nn as nn


def masked_huber_loss(y_true, y_pred, mask, delta=1.0):
    """NumPy version of masked Huber loss."""
    error = y_true - y_pred
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * quadratic ** 2 + delta * linear
    loss = loss * mask
    return np.sum(loss) / np.sum(mask)


class MaskedHuberLoss(nn.Module):
    """
    PyTorch masked Huber loss module.
    
    Combines MSE and MAE with masking for invalid actions.
    Used in DeepStack to handle variable number of actions.
    """
    
    def __init__(self, delta=1.0):
        """
        Initialize masked Huber loss.
        
        Args:
            delta: Threshold for switching between quadratic and linear loss
        """
        super().__init__()
        self.delta = delta
    
    def forward(self, y_pred, y_true, mask):
        """
        Compute masked Huber loss.
        
        Args:
            y_pred: Predicted values [batch_size, num_outputs]
            y_true: Target values [batch_size, num_outputs]
            mask: Binary mask [batch_size, num_outputs]
        
        Returns:
            Scalar loss value
        """
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # Huber loss: quadratic for small errors, linear for large errors
        quadratic = torch.min(abs_error, torch.tensor(self.delta, device=abs_error.device))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        
        # Apply mask and compute mean
        masked_loss = loss * mask
        return torch.sum(masked_loss) / (torch.sum(mask) + 1e-8)

