"""
Value Network for DeepStack Terminal Estimation

Neural network that maps public state + hand ranges to counterfactual values.
Used for terminal node estimation in depth-limited lookahead.

Architecture matches DeepStack paper specification:
- Input: Public state features + normalized range vector
- Output: Counterfactual values for all possible hands
- 4-layer MLP with 12K+ parameters
"""

from typing import Optional, List, Tuple
import os

import numpy as np

# Optional PyTorch import
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ValueNetwork:
    """
    DeepStack-style value network for counterfactual value estimation.
    
    This network is the core of DeepStack's offline component:
    1. Input: Public state + opponent range → Feature vector
    2. Output: Counterfactual values for all player's hands
    
    The network is trained on CFR-solved poker situations to
    approximate terminal values without fully solving to showdown.
    
    Usage:
        net = ValueNetwork(input_size=36, hidden_layers=[256, 128, 64])
        values = net.predict(public_features, opponent_range)
    """
    
    def __init__(
        self,
        input_size: int = 36,
        hidden_layers: List[int] = None,
        output_size: int = 36,
        dropout: float = 0.1,
        device: str = 'auto'
    ):
        """
        Initialize value network.
        
        Args:
            input_size: Number of input features (hand buckets)
            hidden_layers: List of hidden layer sizes
            output_size: Number of output values (hand buckets)
            dropout: Dropout rate for regularization
            device: Computation device ('auto', 'cuda', 'cpu')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [256, 128, 64]
        self.output_size = output_size
        self.dropout = dropout
        
        # Determine device
        if device == 'auto':
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.temperature = 1.0  # For calibration
        
        if TORCH_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Build the neural network."""
        layers = []
        in_features = self.input_size * 2  # Public state + opponent range
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            in_features = hidden_size
        
        layers.append(nn.Linear(in_features, self.output_size))
        
        self.model = nn.Sequential(*layers).to(self.device)
    
    def predict(
        self,
        public_features: np.ndarray,
        opponent_range: np.ndarray
    ) -> np.ndarray:
        """
        Predict counterfactual values.
        
        Args:
            public_features: Public state feature vector (pot, street, board)
            opponent_range: Normalized opponent range distribution
            
        Returns:
            Counterfactual values for all player hands
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback to uniform values
            return np.zeros(self.output_size)
        
        # Combine inputs
        combined = np.concatenate([public_features, opponent_range])
        x = torch.tensor(combined, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            values = self.model(x).squeeze(0)
            values = values / self.temperature  # Apply calibration
        
        return values.cpu().numpy()
    
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """Forward pass for training."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model(x)
    
    def train_step(
        self,
        public_features: np.ndarray,
        opponent_range: np.ndarray,
        target_values: np.ndarray,
        optimizer: 'torch.optim.Optimizer'
    ) -> float:
        """
        Single training step.
        
        Args:
            public_features: Batch of public state features
            opponent_range: Batch of opponent ranges
            target_values: Batch of target CFV values
            optimizer: PyTorch optimizer
            
        Returns:
            Loss value
        """
        if not TORCH_AVAILABLE or self.model is None:
            return 0.0
        
        self.model.train()
        
        # Prepare inputs
        combined = np.concatenate([public_features, opponent_range], axis=1)
        x = torch.tensor(combined, dtype=torch.float32, device=self.device)
        y = torch.tensor(target_values, dtype=torch.float32, device=self.device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = self.model(x)
        loss = F.huber_loss(predictions, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        """Save model weights."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'temperature': self.temperature
        }, path)
        print(f"[ValueNetwork] Saved to {path}")
    
    def load(self, path: str):
        """Load model weights."""
        if not TORCH_AVAILABLE:
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Rebuild model if architecture differs
        if checkpoint.get('hidden_layers') != self.hidden_layers:
            self.hidden_layers = checkpoint['hidden_layers']
            self._build_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.temperature = checkpoint.get('temperature', 1.0)
        self.model.eval()
        print(f"[ValueNetwork] Loaded from {path}")
    
    def calibrate_temperature(
        self,
        val_predictions: np.ndarray,
        val_targets: np.ndarray
    ) -> float:
        """
        Find optimal temperature scaling factor.
        
        Uses grid search to minimize Expected Calibration Error.
        
        Args:
            val_predictions: Model predictions on validation set
            val_targets: Ground truth values
            
        Returns:
            Optimal temperature value
        """
        best_temp = 1.0
        best_ece = float('inf')
        
        for temp in np.linspace(0.5, 2.0, 31):
            scaled = val_predictions / temp
            ece = self._compute_ece(scaled, val_targets)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp
        
        self.temperature = best_temp
        print(f"[ValueNetwork] Calibrated temperature: {best_temp:.3f} (ECE: {best_ece:.4f})")
        return best_temp
    
    def _compute_ece(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        # Simplified ECE for regression
        errors = np.abs(predictions - targets)
        return np.mean(errors)
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        if not TORCH_AVAILABLE or self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def export_onnx(self, path: str, opset_version: int = 14):
        """Export model to ONNX format."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        dummy_input = torch.randn(1, self.input_size * 2, device=self.device)
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['values'],
            dynamic_axes={'input': {0: 'batch_size'}, 'values': {0: 'batch_size'}}
        )
        print(f"[ValueNetwork] Exported ONNX to {path}")


def load_value_network(path: str, device: str = 'auto') -> ValueNetwork:
    """
    Load a trained value network from checkpoint.
    
    Args:
        path: Path to checkpoint file
        device: Computation device
        
    Returns:
        Loaded ValueNetwork instance
    """
    network = ValueNetwork(device=device)
    network.load(path)
    return network


__all__ = ['ValueNetwork', 'load_value_network']
