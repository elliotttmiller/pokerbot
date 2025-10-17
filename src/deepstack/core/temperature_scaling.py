"""
Temperature Scaling for Model Calibration

Implementation of temperature scaling for post-hoc calibration of neural network predictions.
Based on "On Calibration of Modern Neural Networks" (Guo et al., 2017).

Temperature scaling is a simple yet effective method to calibrate model predictions by
learning a single scalar parameter (temperature) that rescales the logits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemperatureScaler:
    """
    Temperature scaling for model calibration.
    
    This learns a single scalar parameter T (temperature) that divides the model outputs
    before computing the final predictions. It improves calibration without changing
    the model's accuracy.
    
    For regression tasks (like DeepStack value prediction), temperature scaling
    adjusts the scale of predictions to match the target distribution better.
    """
    
    def __init__(self):
        """Initialize temperature scaler with default temperature of 1.0."""
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        self.is_fitted = False
        
    def fit(self, predictions, targets, masks=None, lr=0.01, max_iter=50):
        """
        Fit the temperature parameter using validation data.
        
        Args:
            predictions: Model predictions, shape [N, D] or [N]
            targets: Ground truth targets, shape [N, D] or [N]
            masks: Optional binary mask, shape [N, D] or [N]
            lr: Learning rate for optimization
            max_iter: Maximum number of optimization iterations
            
        Returns:
            Fitted temperature value (float)
        """
        # Convert to tensors if needed
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.FloatTensor(predictions)
        if not isinstance(targets, torch.Tensor):
            targets = torch.FloatTensor(targets)
        if masks is not None and not isinstance(masks, torch.Tensor):
            masks = torch.FloatTensor(masks)
            
        # Setup optimizer
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            # Scale predictions by temperature
            scaled_preds = predictions / self.temperature
            
            # Compute MSE loss
            if masks is not None:
                loss = torch.sum(((scaled_preds - targets) ** 2) * masks) / (torch.sum(masks) + 1e-8)
            else:
                loss = torch.mean((scaled_preds - targets) ** 2)
            
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        self.is_fitted = True
        return float(self.temperature.item())
    
    def transform(self, predictions):
        """
        Apply temperature scaling to predictions.
        
        Args:
            predictions: Model predictions to scale
            
        Returns:
            Scaled predictions
        """
        if not self.is_fitted:
            return predictions
            
        if isinstance(predictions, np.ndarray):
            return predictions / float(self.temperature.item())
        elif isinstance(predictions, torch.Tensor):
            return predictions / self.temperature
        else:
            raise TypeError("Predictions must be numpy array or torch tensor")
    
    def get_temperature(self):
        """Get the current temperature value."""
        return float(self.temperature.item())
    
    def save(self, path):
        """Save temperature to file."""
        torch.save({'temperature': self.temperature}, path)
    
    def load(self, path):
        """Load temperature from file."""
        state = torch.load(path, map_location='cpu')
        self.temperature = nn.Parameter(state['temperature'])
        self.is_fitted = True


class PlattScaler:
    """
    Platt scaling for binary classification calibration.
    
    Learns parameters a, b such that calibrated_prob = sigmoid(a * logit + b).
    This is more flexible than temperature scaling but requires more data.
    """
    
    def __init__(self):
        """Initialize Platt scaler."""
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))
        self.is_fitted = False
    
    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """
        Fit Platt scaling parameters.
        
        Args:
            logits: Model logits (before sigmoid)
            labels: True binary labels
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Fitted parameters (a, b)
        """
        if not isinstance(logits, torch.Tensor):
            logits = torch.FloatTensor(logits)
        if not isinstance(labels, torch.Tensor):
            labels = torch.FloatTensor(labels)
        
        optimizer = optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled = self.a * logits + self.b
            probs = torch.sigmoid(scaled)
            loss = nn.functional.binary_cross_entropy(probs, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.is_fitted = True
        return (float(self.a.item()), float(self.b.item()))
    
    def transform(self, logits):
        """Apply Platt scaling to logits."""
        if not self.is_fitted:
            return torch.sigmoid(logits) if isinstance(logits, torch.Tensor) else 1.0 / (1.0 + np.exp(-logits))
        
        if isinstance(logits, np.ndarray):
            scaled = float(self.a.item()) * logits + float(self.b.item())
            return 1.0 / (1.0 + np.exp(-scaled))
        elif isinstance(logits, torch.Tensor):
            scaled = self.a * logits + self.b
            return torch.sigmoid(scaled)
        else:
            raise TypeError("Logits must be numpy array or torch tensor")


def evaluate_calibration(predictions, targets, masks=None, n_bins=10):
    """
    Evaluate model calibration using reliability diagrams.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        masks: Optional binary mask
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with calibration metrics:
        - expected_calibration_error (ECE)
        - maximum_calibration_error (MCE)
        - bin_accuracy: Accuracy per bin
        - bin_confidence: Confidence per bin
        - bin_counts: Number of samples per bin
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if masks is not None and isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    # Apply mask if provided
    if masks is not None:
        valid_mask = masks.flatten() > 0
        predictions = predictions.flatten()[valid_mask]
        targets = targets.flatten()[valid_mask]
    else:
        predictions = predictions.flatten()
        targets = targets.flatten()
    
    # For regression, we adapt ECE by binning absolute errors
    errors = np.abs(predictions - targets)
    pred_magnitude = np.abs(predictions)
    
    # Create bins based on prediction magnitude
    bin_edges = np.linspace(pred_magnitude.min(), pred_magnitude.max(), n_bins + 1)
    bin_indices = np.digitize(pred_magnitude, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_accuracy = []
    bin_confidence = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_mask = bin_indices == i
        if bin_mask.sum() > 0:
            bin_accuracy.append(np.mean(errors[bin_mask]))
            bin_confidence.append(np.mean(pred_magnitude[bin_mask]))
            bin_counts.append(bin_mask.sum())
        else:
            bin_accuracy.append(0.0)
            bin_confidence.append(0.0)
            bin_counts.append(0)
    
    bin_accuracy = np.array(bin_accuracy)
    bin_confidence = np.array(bin_confidence)
    bin_counts = np.array(bin_counts)
    
    # Expected Calibration Error (ECE)
    total_samples = bin_counts.sum()
    ece = np.sum(bin_counts * np.abs(bin_accuracy - bin_confidence)) / max(total_samples, 1)
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(bin_accuracy - bin_confidence))
    
    return {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'bin_accuracy': bin_accuracy.tolist(),
        'bin_confidence': bin_confidence.tolist(),
        'bin_counts': bin_counts.tolist()
    }
