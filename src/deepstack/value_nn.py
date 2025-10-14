"""
ValueNN: DeepStack value network for counterfactual value estimation.

Neural network that estimates counterfactual values at lookahead leaves,
replacing expensive subtree solving. This is the core of DeepStack's
depth-limited search.

Based on the original DeepStack ValueNN module.
"""
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class ValueNetwork(nn.Module):
    """
    Neural network for value estimation.
    
    Architecture:
        Input: [player_range, opponent_range, pot_size]
        Hidden: Multiple layers with PReLU activation
        Output: [player_values, opponent_values]
    """
    
    def __init__(self, num_hands: int = 169, hidden_sizes: list = None):
        """
        Initialize value network.
        
        Args:
            num_hands: Number of hand buckets (169 for Hold'em, 6 for Leduc)
            hidden_sizes: List of hidden layer sizes (default: [512, 512, 512, 512])
        """
        super().__init__()
        
        self.num_hands = num_hands
        if hidden_sizes is None:
            if num_hands <= 6:
                # Leduc: smaller network
                hidden_sizes = [50, 50, 50]
            else:
                # Hold'em: larger network
                hidden_sizes = [512, 512, 512, 512]
        
        # Input: 2 * num_hands (two ranges) + 1 (pot size)
        input_size = 2 * num_hands + 1
        
        # Output: 2 * num_hands (values for each player's hands)
        output_size = 2 * num_hands
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.PReLU())  # Parametric ReLU
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor [batch_size, 2*num_hands + 1]
               Format: [player_range, opponent_range, pot_size]
            
        Returns:
            Output tensor [batch_size, 2*num_hands]
            Format: [player_values, opponent_values]
        """
        # Extract ranges from input
        range_vectors = x[:, :2 * self.num_hands]
        
        # Network forward pass
        output = self.network(x)
        
        # Add residual connection based on range overlap
        # This enforces value symmetry: value(P1,P2) + value(P2,P1) = 0
        dot_product = (range_vectors[:, :self.num_hands] * 
                      range_vectors[:, self.num_hands:]).sum(dim=1, keepdim=True)
        residual = -0.5 * dot_product
        residual = residual.expand(-1, 2 * self.num_hands)
        
        return output + residual


class ValueNN:
    """
    Wrapper for value network with loading/saving functionality.
    
    Provides interface compatible with original DeepStack ValueNN.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 num_hands: int = 169,
                 hidden_sizes: list = None,
                 device: str = 'cpu'):
        """
        Initialize ValueNN wrapper.
        
        Args:
            model_path: Path to saved model file
            num_hands: Number of hand buckets
            hidden_sizes: Hidden layer sizes
            device: 'cpu' or 'cuda'
        """
        self.num_hands = num_hands
        self.device = torch.device(device)
        
        # Create network
        self.network = ValueNetwork(num_hands, hidden_sizes).to(self.device)
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        
        # Set to evaluation mode by default
        self.network.eval()
    
    def get_value(self, inputs: np.ndarray) -> np.ndarray:
        """
        Get neural network output for batch of inputs.
        
        Args:
            inputs: Input array [N, 2*num_hands + 1]
                   Each row: [player_range, opponent_range, pot_size]
            
        Returns:
            Output array [N, 2*num_hands]
            Each row: [player_values, opponent_values]
        """
        # Convert to tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()
        
        inputs = inputs.to(self.device)
        
        # Forward pass (no gradient)
        with torch.no_grad():
            outputs = self.network(inputs)
        
        # Convert back to numpy
        return outputs.cpu().numpy()
    
    def get_value_single(self, player_range: np.ndarray, 
                         opponent_range: np.ndarray,
                         pot_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get values for single game state.
        
        Args:
            player_range: Player's range [num_hands]
            opponent_range: Opponent's range [num_hands]
            pot_size: Normalized pot size
            
        Returns:
            Tuple of (player_values, opponent_values)
        """
        # Construct input
        input_vec = np.concatenate([player_range, opponent_range, [pot_size]])
        input_batch = input_vec.reshape(1, -1)
        
        # Get output
        output = self.get_value(input_batch)[0]
        
        # Split into player and opponent values
        player_values = output[:self.num_hands]
        opponent_values = output[self.num_hands:]
        
        return player_values, opponent_values
    
    def load(self, model_path: str):
        """
        Load model from file.
        
        Args:
            model_path: Path to model file (.pt or .pth)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        self.network.load_state_dict(state_dict)
        self.network.eval()
        
        print(f"Loaded value network from {model_path}")
    
    def save(self, model_path: str):
        """
        Save model to file.
        
        Args:
            model_path: Path to save model (.pt or .pth)
        """
        # Create directory if needed
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save state dict
        torch.save(self.network.state_dict(), model_path)
        print(f"Saved value network to {model_path}")
    
    def set_training_mode(self, training: bool = True):
        """
        Set network to training or evaluation mode.
        
        Args:
            training: True for training mode, False for eval mode
        """
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def get_parameters(self):
        """Get network parameters for optimization."""
        return self.network.parameters()
    
    def to_device(self, device: str):
        """
        Move network to device.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.network = self.network.to(self.device)
