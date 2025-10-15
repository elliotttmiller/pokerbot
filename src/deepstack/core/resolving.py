"""
Resolving: DeepStack continual re-solving API.

High-level interface for depth-limited lookahead solving during gameplay.
Wraps Lookahead class and provides methods for re-solving at each decision point.

Based on the original DeepStack Resolving module.
"""
import numpy as np
from typing import List, Optional, Tuple, Dict
from deepstack.core.tree_builder import PokerTreeBuilder, PokerTreeNode
from deepstack.core.tree_cfr import TreeCFR
from deepstack.core.terminal_equity import TerminalEquity
from deepstack.core.cfrd_gadget import CFRDGadget
from deepstack.core.value_nn import ValueNN


class Resolving:
    """
    Continual re-solving for DeepStack decision-making.
    
    Provides the main API for:
    1. Re-solving at root node with fixed ranges
    2. Re-solving at subsequent nodes with CFRDGadget
    3. Extracting strategies and counterfactual values
    
    This class is used by the player during actual gameplay.
    """
    
    def __init__(self, num_hands: int = 6, game_variant: str = 'leduc',
                 value_nn: Optional[ValueNN] = None):
        """
        Initialize resolving module.
        
        Args:
            num_hands: Number of possible hands
            game_variant: 'leduc' or 'holdem'
            value_nn: Pre-trained value network (optional)
        """
        self.num_hands = num_hands
        self.game_variant = game_variant
        self.value_nn = value_nn
        
        # Storage for last solve results
        self.current_node = None
        self.player_range = None
        self.opponent_range = None
        self.strategy = None
        self.root_cfv = None
        self.root_cfv_both = None
        self.action_cfvs = {}
        
        # Components
        self.tree_builder = PokerTreeBuilder(game_variant=game_variant)
        self.terminal_equity = TerminalEquity(game_variant=game_variant, 
                                             num_hands=num_hands)
    
    def resolve_first_node(self, node: Dict, player_range: np.ndarray, 
                           opponent_range: np.ndarray, iterations: int = 1000):
        """
        Re-solve depth-limited lookahead at root node.
        
        Used at the start of the game or betting round when ranges are fixed.
        
        Args:
            node: Node parameters (street, bets, board, current_player)
            player_range: Player's range (probability distribution)
            opponent_range: Opponent's range (probability distribution)
            iterations: Number of CFR iterations
        """
        # Store inputs
        self.player_range = player_range
        self.opponent_range = opponent_range
        
        # Build lookahead tree from node
        tree_params = {
            'street': node.get('street', 0),
            'bets': node.get('bets', [20, 20]),
            'current_player': node.get('current_player', 1),
            'board': node.get('board', []),
            'limit_to_street': node.get('limit_to_street', True),
            'bet_sizing': node.get('bet_sizing', [1.0])
        }
        
        self.current_node = self.tree_builder.build_tree(tree_params)
        
        # Set board for terminal equity
        self.terminal_equity.set_board(tree_params['board'])
        
        # Run CFR on tree
        cfr_solver = TreeCFR(skip_iterations=200)
        starting_ranges = np.array([player_range, opponent_range])
        
        result = cfr_solver.run_cfr(self.current_node, starting_ranges, iterations)
        
        # Extract strategy and values
        self.strategy = result['strategy']
        self._compute_root_values(self.current_node, player_range, opponent_range)
        self._compute_action_cfvs(self.current_node, player_range, opponent_range)
    
    def resolve(self, node: Dict, player_range: np.ndarray, 
                opponent_cfvs: np.ndarray, iterations: int = 1000):
        """
        Re-solve using player range and CFRDGadget for opponent.
        
        Used during gameplay after opponent has acted. Opponent's range
        is reconstructed from their counterfactual values.
        
        Args:
            node: Node parameters
            player_range: Player's updated range
            opponent_cfvs: Opponent's CFVs from previous solve
            iterations: Number of CFR iterations
        """
        # Store inputs
        self.player_range = player_range
        
        # Reconstruct opponent range using gadget
        board = node.get('board', [])
        gadget = CFRDGadget(board, player_range, opponent_cfvs)
        opponent_range = gadget.compute_opponent_range(opponent_cfvs, 0)
        self.opponent_range = opponent_range
        
        # Build and solve tree (same as resolve_first_node)
        tree_params = {
            'street': node.get('street', 0),
            'bets': node.get('bets', [20, 20]),
            'current_player': node.get('current_player', 1),
            'board': board,
            'limit_to_street': node.get('limit_to_street', True),
            'bet_sizing': node.get('bet_sizing', [1.0])
        }
        
        self.current_node = self.tree_builder.build_tree(tree_params)
        self.terminal_equity.set_board(board)
        
        # Run CFR
        cfr_solver = TreeCFR(skip_iterations=200)
        starting_ranges = np.array([player_range, opponent_range])
        
        result = cfr_solver.run_cfr(self.current_node, starting_ranges, iterations)
        
        # Extract results
        self.strategy = result['strategy']
        self._compute_root_values(self.current_node, player_range, opponent_range)
        self._compute_action_cfvs(self.current_node, player_range, opponent_range)
    
    def get_possible_actions(self) -> List[str]:
        """
        Get list of possible actions at current node.
        
        Returns:
            List of action strings (e.g., ['fold', 'call', 'raise_1.0'])
        """
        if self.current_node is None or not self.current_node.children:
            return []
        return self.current_node.actions

    def get_strategy(self) -> Dict[str, np.ndarray]:
        """Return the latest average strategy dictionary."""
        if self.strategy is None:
            return {}
        return self.strategy

    def get_root_strategy(self) -> Optional[np.ndarray]:
        """Return strategy probabilities for the root information set."""
        if self.strategy is None:
            return None
        return self.strategy.get('root')
    
    def get_root_cfv(self) -> np.ndarray:
        """
        Get average counterfactual values for re-solve player at root.
        
        Returns:
            CFV vector for player's range
        """
        return self.root_cfv if self.root_cfv is not None else np.zeros(self.num_hands)
    
    def get_root_cfv_both_players(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get average counterfactual values for both players at root.
        
        Returns:
            Tuple of (player_cfvs, opponent_cfvs)
        """
        if self.root_cfv_both is None:
            return (np.zeros(self.num_hands), np.zeros(self.num_hands))
        return self.root_cfv_both
    
    def get_action_cfv(self, action: str) -> np.ndarray:
        """
        Get opponent CFVs after player takes action.
        
        Args:
            action: Action string (e.g., 'call', 'raise_1.0')
            
        Returns:
            Opponent's CFV vector
        """
        return self.action_cfvs.get(action, np.zeros(self.num_hands))
    
    def get_chance_action_cfv(self, action: int, board: List) -> np.ndarray:
        """
        Get opponent CFVs after chance event (board cards dealt).
        
        Args:
            action: Chance action index
            board: New board cards
            
        Returns:
            Opponent's CFV vector
        """
        # After chance event, re-evaluate with new board
        self.terminal_equity.set_board(board)
        
        # Simplified: return average CFV based on equity
        if self.player_range is not None and self.opponent_range is not None:
            ev = self.terminal_equity.compute_expected_value(
                self.player_range, self.opponent_range, pot_size=100.0
            )
            return ev
        
        return np.zeros(self.num_hands)
    
    def get_action_strategy(self, action: str) -> float:
        """
        Get probability of taking action in current strategy.
        
        Args:
            action: Action string
            
        Returns:
            Probability of action (0.0 to 1.0)
        """
        if self.strategy is None or 'root' not in self.strategy:
            # No strategy computed, return uniform
            num_actions = len(self.get_possible_actions())
            return 1.0 / num_actions if num_actions > 0 else 0.0
        
        # Get strategy at root node
        root_strategy = self.strategy.get('root', None)
        if root_strategy is None:
            num_actions = len(self.get_possible_actions())
            return 1.0 / num_actions if num_actions > 0 else 0.0
        
        # Find action index
        actions = self.get_possible_actions()
        if action in actions:
            action_idx = actions.index(action)
            if action_idx < len(root_strategy):
                return root_strategy[action_idx]
        
        return 0.0
    
    def _compute_root_values(self, node: PokerTreeNode, 
                             player_range: np.ndarray,
                             opponent_range: np.ndarray):
        """
        Compute counterfactual values at root node.
        
        Args:
            node: Root node
            player_range: Player's range
            opponent_range: Opponent's range
        """
        # Compute expected values using terminal equity
        player_ev = self.terminal_equity.compute_expected_value(
            player_range, opponent_range, pot_size=node.pot
        )
        
        opponent_ev = self.terminal_equity.compute_expected_value(
            opponent_range, player_range, pot_size=node.pot
        )
        
        self.root_cfv = player_ev
        self.root_cfv_both = (player_ev, opponent_ev)
    
    def _compute_action_cfvs(self, node: PokerTreeNode,
                             player_range: np.ndarray,
                             opponent_range: np.ndarray):
        """
        Compute opponent CFVs for each action.
        
        Args:
            node: Current node
            player_range: Player's range
            opponent_range: Opponent's range
        """
        self.action_cfvs = {}
        
        for action in node.actions:
            # Simplified: compute CFV based on equity
            cfv = self.terminal_equity.compute_expected_value(
                opponent_range, player_range, pot_size=node.pot
            )
            self.action_cfvs[action] = cfv
