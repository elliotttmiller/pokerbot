"""
Value Network - The "Intuition" of the Strategist.

Provides a fast, accurate estimation of the expected value (EV) of any 
game state. This allows the CFR search to be shallow but powerful.

Architecture based on DeepStack paper:
- Input: Vectorized game state (ranges + pot size)
- Output: Predicted EV for each hand in range
- Training: Self-play with CFR-generated data

References:
- DeepStack paper Section S3: Neural network architecture
- DeepStack-Leduc: Reference implementation
"""

import os
import numpy as np
from typing import Optional, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ValueNetwork(nn.Module):
    """
    Neural network for counterfactual value estimation.
    
    Architecture per DeepStack paper (Section S3):
    - Input: [player_range, opponent_range, pot_size]
    - Hidden: 7 layers × 500 units with PReLU activation (Hold'em)
    - Output: [player_values, opponent_values]
    
    The network learns to estimate counterfactual values at the leaves
    of the lookahead tree, replacing expensive full tree solving.
    """
    
    def __init__(self,
                 num_buckets: int = 169,
                 hidden_sizes: Optional[List[int]] = None,
                 use_batch_norm: bool = True):
        """
        Initialize value network.
        
        Args:
            num_buckets: Number of hand buckets (169 for Hold'em preflop)
            hidden_sizes: List of hidden layer sizes
            use_batch_norm: Use batch normalization (per paper)
        """
        super().__init__()
        
        self.num_buckets = num_buckets
        
        # Default hidden sizes per DeepStack paper
        if hidden_sizes is None:
            if num_buckets <= 10:
                # Leduc-style game: smaller network
                hidden_sizes = [64, 64, 64]
            else:
                # Hold'em: 7 layers × 500 units (Table S2)
                hidden_sizes = [500, 500, 500, 500, 500, 500, 500]
        
        # Input: 2 * buckets (ranges) + 1 (pot)
        input_size = 2 * num_buckets + 1
        # Output: 2 * buckets (values for both players)
        output_size = 2 * num_buckets
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.PReLU())  # Parametric ReLU per paper
            prev_size = hidden_size
        
        # Output layer (no activation)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, 2*num_buckets + 1]
               Format: [player_range, opponent_range, pot_size]
               
        Returns:
            Output tensor [batch_size, 2*num_buckets]
            Format: [player_values, opponent_values]
        """
        return self.network(x)
    
    def get_values(self, player_range: np.ndarray, 
                   opponent_range: np.ndarray,
                   pot_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get counterfactual values for a single game state.
        
        Args:
            player_range: Player's range distribution [num_buckets]
            opponent_range: Opponent's range distribution [num_buckets]
            pot_size: Normalized pot size
            
        Returns:
            Tuple of (player_values, opponent_values)
        """
        # Build input vector
        input_vec = np.concatenate([player_range, opponent_range, [pot_size]])
        input_tensor = torch.from_numpy(input_vec).float().unsqueeze(0)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            output = self.network(input_tensor)[0].numpy()
        
        # Split output
        player_values = output[:self.num_buckets]
        opponent_values = output[self.num_buckets:]
        
        return player_values, opponent_values


class ValueNetworkWrapper:
    """
    High-level wrapper for ValueNetwork with loading/saving.
    
    Provides the interface expected by the CFR solver.
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 num_buckets: int = 169,
                 device: str = 'cpu'):
        """
        Initialize wrapper.
        
        Args:
            model_path: Path to saved model weights
            num_buckets: Number of hand buckets
            device: Computation device
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for ValueNetwork")
        
        self.num_buckets = num_buckets
        self.device = torch.device(device)
        
        # Create network
        self.network = ValueNetwork(num_buckets).to(self.device)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        
        self.network.eval()
    
    def get_value(self, player_range: np.ndarray,
                  opponent_range: np.ndarray,
                  pot_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get counterfactual values for game state.
        
        Args:
            player_range: Player's range [num_buckets]
            opponent_range: Opponent's range [num_buckets]
            pot_size: Normalized pot size
            
        Returns:
            Tuple of (player_cfv, opponent_cfv)
        """
        # Build input
        input_vec = np.concatenate([player_range, opponent_range, [pot_size]])
        input_tensor = torch.from_numpy(input_vec).float().unsqueeze(0).to(self.device)
        
        # Forward
        with torch.no_grad():
            output = self.network(input_tensor)[0].cpu().numpy()
        
        return output[:self.num_buckets], output[self.num_buckets:]
    
    def get_batch_values(self, inputs: np.ndarray) -> np.ndarray:
        """
        Batch inference for multiple states.
        
        Args:
            inputs: Input array [batch_size, 2*num_buckets + 1]
            
        Returns:
            Output array [batch_size, 2*num_buckets]
        """
        input_tensor = torch.from_numpy(inputs).float().to(self.device)
        
        with torch.no_grad():
            output = self.network(input_tensor).cpu().numpy()
        
        return output
    
    def load(self, path: str):
        """Load model weights."""
        state_dict = torch.load(path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        print(f"[ValueNetwork] Loaded from {path}")
    
    def save(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)
        print(f"[ValueNetwork] Saved to {path}")
    
    def train_mode(self, training: bool = True):
        """Set training or evaluation mode."""
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def get_parameters(self):
        """Get network parameters for optimizer."""
        return self.network.parameters()


__all__ = ['ValueNetwork', 'ValueNetworkWrapper']
