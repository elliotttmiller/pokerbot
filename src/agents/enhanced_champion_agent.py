"""
Enhanced Champion Agent with DeepStack Value Network Integration

This module integrates advanced components from the poker-ai repository:
1. DeepStack value network for counterfactual value estimation
2. CFR+ algorithm with pruning and LCFR
3. Enhanced training capabilities

The enhanced agent maintains backward compatibility while adding
championship-level components.
"""

from typing import Optional
import numpy as np

from .champion_agent import ChampionAgent
from .cfr_plus_agent import CFRPlusAgent

# Try to import DeepStack components
try:
    from .deepstack_value_network import DeepStackValueNetwork, build_deepstack_network
    DEEPSTACK_AVAILABLE = True
except ImportError:
    DEEPSTACK_AVAILABLE = False


class EnhancedChampionAgent(ChampionAgent):
    """
    Enhanced Champion Agent with DeepStack and CFR+ integration.
    
    New capabilities:
    - DeepStack value network for better value estimation
    - CFR+ with pruning for faster convergence
    - Range-based decision making
    - Continual re-solving support (future)
    """
    
    def __init__(
        self,
        name: str = "EnhancedChampion",
        use_deepstack: bool = True,
        use_cfr_plus: bool = True,
        deepstack_architecture: str = 'medium',
        **kwargs
    ):
        """
        Initialize Enhanced Champion Agent.
        
        Args:
            name: Agent name
            use_deepstack: Whether to use DeepStack value network
            use_cfr_plus: Whether to use CFR+ instead of vanilla CFR
            deepstack_architecture: Size of DeepStack network ('small', 'medium', 'large')
            **kwargs: Additional arguments for base ChampionAgent
        """
        # Initialize base champion agent
        super().__init__(name=name, **kwargs)
        
        # Replace CFR with CFR+ if requested
        if use_cfr_plus:
            self.cfr = CFRPlusAgent(name=f"{name}_CFRPlus")
            print(f"  ✓ Using CFR+ with pruning and LCFR")
        
        # Initialize DeepStack value network
        self.value_network = None
        self.use_deepstack = use_deepstack and DEEPSTACK_AVAILABLE
        
        if self.use_deepstack:
            try:
                self.value_network = build_deepstack_network(
                    bucket_count=169,  # Standard for Texas Hold'em
                    architecture=deepstack_architecture
                )
                print(f"  ✓ DeepStack value network initialized ({deepstack_architecture})")
            except Exception as e:
                print(f"  ⚠ Could not initialize DeepStack network: {e}")
                self.use_deepstack = False
        elif use_deepstack and not DEEPSTACK_AVAILABLE:
            print(f"  ⚠ DeepStack requested but PyTorch not available")
    
    def estimate_hand_values(
        self,
        my_range: np.ndarray,
        opponent_range: np.ndarray,
        pot_size: float
    ) -> tuple:
        """
        Estimate counterfactual values using DeepStack network.
        
        Args:
            my_range: Probability distribution over my possible hands
            opponent_range: Probability distribution over opponent's hands
            pot_size: Current pot size (normalized)
        
        Returns:
            Tuple of (my_values, opponent_values)
        """
        if not self.use_deepstack or self.value_network is None:
            # Fallback to uniform values
            return (
                np.ones(len(my_range)) * pot_size / 2,
                np.ones(len(opponent_range)) * pot_size / 2
            )
        
        try:
            import torch
            
            # Convert to torch tensors
            my_range_tensor = torch.tensor(my_range, dtype=torch.float32)
            opp_range_tensor = torch.tensor(opponent_range, dtype=torch.float32)
            
            # Get value estimates
            my_values, opp_values = self.value_network.predict_values(
                my_range_tensor,
                opp_range_tensor,
                pot_size
            )
            
            return my_values.numpy(), opp_values.numpy()
        
        except Exception as e:
            print(f"Warning: DeepStack value estimation failed: {e}")
            # Fallback to uniform
            return (
                np.ones(len(my_range)) * pot_size / 2,
                np.ones(len(opponent_range)) * pot_size / 2
            )
    
    def train_cfr_plus(self, num_iterations: int = 1000):
        """
        Train using enhanced CFR+ algorithm.
        
        Args:
            num_iterations: Number of CFR+ iterations
        """
        if isinstance(self.cfr, CFRPlusAgent):
            self.cfr.train(num_iterations)
            
            # Print statistics
            stats = self.cfr.get_training_stats()
            print(f"\nCFR+ Training Statistics:")
            print(f"  Iterations: {stats['iterations']}")
            print(f"  Information sets: {stats['infosets']}")
            print(f"  Average regret: {stats['average_regret']:.6f}")
            print(f"  Pruned actions: {stats['pruned_actions']}/{stats['total_actions']} ({stats['prune_percentage']:.1f}%)")
        else:
            # Fall back to standard CFR training
            super().train_cfr(num_iterations)
    
    def get_enhanced_stats(self) -> dict:
        """
        Get comprehensive agent statistics including enhanced components.
        
        Returns:
            Dictionary with agent statistics
        """
        stats = {
            'name': self.name,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_mode': self.training_mode,
            'use_deepstack': self.use_deepstack,
            'use_cfr_plus': isinstance(self.cfr, CFRPlusAgent),
        }
        
        # Add CFR stats
        if hasattr(self.cfr, 'get_training_stats'):
            stats['cfr'] = self.cfr.get_training_stats()
        else:
            stats['cfr'] = {
                'iterations': self.cfr.iterations,
                'infosets': len(self.cfr.infosets)
            }
        
        # Add DeepStack stats
        if self.value_network is not None:
            try:
                import torch
                stats['deepstack'] = {
                    'parameters': sum(p.numel() for p in self.value_network.parameters()),
                    'bucket_count': self.value_network.bucket_count
                }
            except:
                pass
        
        return stats
    
    def train_value_network(
        self,
        training_data: tuple,
        epochs: int = 100,
        batch_size: int = 128,
        learning_rate: float = 0.001
    ):
        """
        Train the DeepStack value network on counterfactual value data.
        
        Args:
            training_data: Tuple of (inputs, targets) where:
                - inputs: (N, 2*M+1) array of range vectors + pot sizes
                - targets: (N, 2*M) array of counterfactual values
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        
        Returns:
            Training history (list of losses)
        """
        if not self.use_deepstack or self.value_network is None:
            print("Warning: DeepStack value network not available")
            return []
        
        try:
            import torch
            import torch.optim as optim
            from .deepstack_value_network import MaskedHuberLoss
            
            X_train, y_train = training_data
            
            # Convert to tensors
            X_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_tensor = torch.tensor(y_train, dtype=torch.float32)
            
            # Setup training
            optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
            loss_fn = MaskedHuberLoss(delta=1.0)
            
            # Training loop
            history = []
            print(f"\nTraining DeepStack value network...")
            print(f"  Samples: {len(X_train):,}")
            print(f"  Epochs: {epochs}")
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                # Mini-batch training
                for i in range(0, len(X_tensor), batch_size):
                    batch_X = X_tensor[i:i+batch_size]
                    batch_y = y_tensor[i:i+batch_size]
                    
                    # Forward pass
                    optimizer.zero_grad()
                    predictions = self.value_network(batch_X)
                    loss = loss_fn(predictions, batch_y)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                history.append(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
            
            print(f"✓ Value network training complete")
            return history
        
        except Exception as e:
            print(f"Error training value network: {e}")
            import traceback
            traceback.print_exc()
            return []


def create_enhanced_champion(
    name: str = "EnhancedChampion",
    **kwargs
) -> EnhancedChampionAgent:
    """
    Factory function to create an enhanced champion agent.
    
    Args:
        name: Agent name
        **kwargs: Additional arguments for EnhancedChampionAgent
    
    Returns:
        Initialized enhanced champion agent
    """
    return EnhancedChampionAgent(name=name, **kwargs)
