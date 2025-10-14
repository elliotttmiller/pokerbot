"""
DeepStack Value Network Implementation

Implements the neural network architecture from DeepStack-Leduc for predicting
counterfactual values. This network estimates the value of each hand at a given
game state, enabling continual re-solving during gameplay.

Based on: https://github.com/lifrordi/DeepStack-Leduc
Integrated from: https://github.com/elliotttmiller/poker-ai

Architecture:
- Input: Range vectors for both players + pot size (2*M + 1 dimensions)
- Output: Counterfactual values for both players (2*M dimensions)
- Uses residual connections and specialized loss function
"""

import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DeepStack value network disabled.")


class DeepStackValueNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Value network for predicting counterfactual values in poker.
    
    This network takes as input:
    - Player 1 range vector (M dimensions - probability distribution over hands)
    - Player 2 range vector (M dimensions)  
    - Pot size (1 dimension - normalized)
    
    And outputs:
    - Player 1 counterfactual values (M dimensions)
    - Player 2 counterfactual values (M dimensions)
    
    The architecture includes a residual connection that adds -0.5 * dot_product(ranges)
    to encourage the network to learn deviations from uniform strategy.
    """
    
    def __init__(
        self,
        bucket_count: int = 169,  # Number of hand buckets (169 for Texas Hold'em)
        hidden_layers: list = None,  # Hidden layer sizes
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize DeepStack value network.
        
        Args:
            bucket_count: Number of hand buckets/abstractions
            hidden_layers: List of hidden layer sizes (default: [512, 512, 512])
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DeepStack value network")
        
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [512, 512, 512]
        
        self.bucket_count = bucket_count
        self.player_count = 2
        self.output_size = bucket_count * self.player_count
        self.input_size = self.output_size + 1  # ranges + pot size
        
        # Build feedforward network
        layers = []
        prev_size = self.input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, self.output_size))
        
        self.feedforward = nn.Sequential(*layers)
        
        print(f"DeepStackValueNetwork initialized:")
        print(f"  Bucket count: {bucket_count}")
        print(f"  Input size: {self.input_size}")
        print(f"  Output size: {self.output_size}")
        print(f"  Hidden layers: {hidden_layers}")
        print(f"  Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x):
        """
        Forward pass through the value network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
               Format: [p1_range (M), p2_range (M), pot_size (1)]
        
        Returns:
            Tensor of shape (batch_size, output_size)
            Format: [p1_cfvs (M), p2_cfvs (M)]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Extract range vectors (first output_size elements)
        range_vectors = x[:, :self.output_size]
        
        # Feedforward pass
        ff_output = self.feedforward(x)
        
        # Compute residual term: -0.5 * dot_product(ranges)
        # This encourages learning deviations from uniform strategy
        dot_product = (range_vectors * range_vectors).sum(dim=1, keepdim=True)
        residual = -0.5 * dot_product
        
        # Replicate residual across all output dimensions
        residual = residual.expand(-1, self.output_size)
        
        # Add residual connection
        output = ff_output + residual
        
        return output
    
    def predict_values(
        self,
        p1_range,
        p2_range,
        pot_size: float
    ) -> Tuple:
        """
        Predict counterfactual values for both players.
        
        Args:
            p1_range: Player 1 range vector (bucket_count,)
            p2_range: Player 2 range vector (bucket_count,)
            pot_size: Pot size (normalized)
        
        Returns:
            Tuple of (p1_cfvs, p2_cfvs)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Prepare input
        pot_tensor = torch.tensor([pot_size], dtype=torch.float32)
        input_tensor = torch.cat([p1_range, p2_range, pot_tensor])
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        with torch.no_grad():
            output = self.forward(input_tensor)
        
        # Split output into player CFVs
        output = output.squeeze(0)
        p1_cfvs = output[:self.bucket_count]
        p2_cfvs = output[self.bucket_count:]
        
        return p1_cfvs, p2_cfvs


class MaskedHuberLoss(nn.Module if TORCH_AVAILABLE else object):
    """
    Masked Huber loss for training value networks.
    
    The mask allows training only on valid hands (non-zero probability in range).
    Huber loss is more robust to outliers than MSE.
    """
    
    def __init__(self, delta: float = 1.0):
        """
        Initialize masked Huber loss.
        
        Args:
            delta: Threshold for switching between L1 and L2 loss
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for MaskedHuberLoss")
        
        super().__init__()
        self.delta = delta
    
    def forward(
        self,
        predictions,
        targets,
        mask: Optional = None
    ):
        """
        Compute masked Huber loss.
        
        Args:
            predictions: Predicted values (batch_size, output_size)
            targets: Target values (batch_size, output_size)
            mask: Binary mask (batch_size, output_size), 1 for valid, 0 for invalid
        
        Returns:
            Scalar loss value
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Compute Huber loss
        error = predictions - targets
        abs_error = torch.abs(error)
        
        # Huber loss formula
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            # Average only over valid (masked) elements
            return loss.sum() / mask.sum().clamp(min=1.0)
        else:
            return loss.mean()


def build_deepstack_network(
    bucket_count: int = 169,
    architecture: str = 'medium',
    **kwargs
) -> 'DeepStackValueNetwork':
    """
    Build a DeepStack value network with preset architecture.
    
    Args:
        bucket_count: Number of hand buckets
        architecture: Architecture size ('small', 'medium', 'large')
        **kwargs: Additional arguments for network
    
    Returns:
        Initialized DeepStack value network
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for DeepStack value network")
    
    architectures = {
        'small': [256, 256],
        'medium': [512, 512, 512],
        'large': [1024, 1024, 1024, 512]
    }
    
    hidden_layers = architectures.get(architecture, architectures['medium'])
    
    return DeepStackValueNetwork(
        bucket_count=bucket_count,
        hidden_layers=hidden_layers,
        **kwargs
    )
