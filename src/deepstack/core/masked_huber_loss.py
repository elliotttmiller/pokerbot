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
    
    def __init__(self, delta=1.0, normalize_by_valid_fraction: bool = True):
        """
        Initialize masked Huber loss.
        
        Args:
            delta: Threshold for switching between quadratic and linear loss
        """
        super().__init__()
        self.delta = delta
        self.normalize_by_valid_fraction = normalize_by_valid_fraction
    
    def forward(self, y_pred, y_true, mask, sample_weights: torch.Tensor | None = None):
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
        denom = torch.sum(mask) + 1e-8
        if self.normalize_by_valid_fraction:
            # Normalize by fraction of valid outputs to keep gradient scale stable across streets
            B, D = mask.shape
            valid_per_sample = torch.sum(mask, dim=1) + 1e-8
            frac = valid_per_sample / float(D)
            # Weight each sample loss by 1/frac to maintain consistent scale
            sample_loss = torch.sum(masked_loss, dim=1) / valid_per_sample
            scaled = sample_loss / (frac + 1e-8)
            if sample_weights is not None:
                # Broadcast-safe multiply
                scaled = scaled * sample_weights.view(-1)
            return torch.mean(scaled)
        else:
            # Mean over batch optionally weighted
            sample_loss = torch.sum(masked_loss, dim=1) / (torch.sum(mask, dim=1) + 1e-8)
            if sample_weights is not None:
                sample_loss = sample_loss * sample_weights.view(-1)
            return torch.mean(sample_loss)

