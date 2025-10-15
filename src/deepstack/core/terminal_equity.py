"""
TerminalEquity: DeepStack-style terminal equity evaluation for showdown and fold nodes.

Computes equity matrices and counterfactual values at terminal nodes.
Based on the original DeepStack TerminalEquity module.
"""
import numpy as np
from typing import List, Optional


class TerminalEquity:
    """
    Computes equity and counterfactual values at terminal nodes.
    
    Terminal nodes are either:
    1. Call nodes (showdown) - evaluate hand strength
    2. Fold nodes - player folds, opponent wins pot
    """
    
    def __init__(self, game_variant: str = 'leduc', num_hands: int = 6):
        """
        Initialize TerminalEquity calculator.
        
        Args:
            game_variant: 'leduc' for Leduc Hold'em, 'holdem' for Texas Hold'em
            num_hands: Number of possible hands in the game
        """
        self.game_variant = game_variant
        self.num_hands = num_hands
        self.board = None
        self.call_matrix = None
        self.fold_matrix = None

    def set_board(self, board: List):
        """
        Set board cards and compute equity matrices.
        
        Args:
            board: List of board cards
        """
        self.board = board
        self._compute_equity_matrices()

    def _compute_equity_matrices(self):
        """
        Compute call and fold matrices for current board.
        
        Call matrix[i][j] = equity of hand i vs hand j at showdown
        Fold matrix used for fold node payoffs
        """
        self.call_matrix = np.zeros((self.num_hands, self.num_hands))
        self.fold_matrix = np.ones((self.num_hands, self.num_hands))
        
        if self.game_variant == 'leduc':
            self._compute_leduc_equity()
        else:
            self._compute_holdem_equity()

    def _compute_leduc_equity(self):
        """
        Compute Leduc Hold'em equity.
        
        Leduc rules:
        - 6 cards: J♠, J♥, Q♠, Q♥, K♠, K♥
        - Pair beats high card
        - Higher pair beats lower pair
        """
        # Map hands to ranks (0=Jack, 1=Queen, 2=King)
        for i in range(self.num_hands):
            rank_i = i // 2  # Each rank has 2 suits
            
            # Check if hand i makes a pair with board
            pair_i = False
            if self.board:
                board_rank = self._get_card_rank(self.board[0]) if self.board else -1
                pair_i = (rank_i == board_rank)
            
            for j in range(self.num_hands):
                if i == j:
                    # Same hand - impossible
                    self.call_matrix[i][j] = 0.0
                    continue
                
                rank_j = j // 2
                
                # Check if hand j makes a pair
                pair_j = False
                if self.board:
                    pair_j = (rank_j == board_rank)
                
                # Determine winner
                if pair_i and not pair_j:
                    # Player has pair, opponent doesn't
                    self.call_matrix[i][j] = 1.0
                elif not pair_i and pair_j:
                    # Opponent has pair, player doesn't
                    self.call_matrix[i][j] = 0.0
                elif pair_i and pair_j:
                    # Both have pairs - compare ranks
                    if rank_i > rank_j:
                        self.call_matrix[i][j] = 1.0
                    elif rank_i < rank_j:
                        self.call_matrix[i][j] = 0.0
                    else:
                        self.call_matrix[i][j] = 0.5  # Tie
                else:
                    # Neither has pair - compare high cards
                    if rank_i > rank_j:
                        self.call_matrix[i][j] = 1.0
                    elif rank_i < rank_j:
                        self.call_matrix[i][j] = 0.0
                    else:
                        self.call_matrix[i][j] = 0.5  # Tie

    def _compute_holdem_equity(self):
        """
        Compute Texas Hold'em equity (simplified).
        
        Uses hand strength approximation based on bucket indices.
        In production, would use Monte Carlo simulation or lookup tables.
        """
        for i in range(self.num_hands):
            for j in range(self.num_hands):
                if i == j:
                    self.call_matrix[i][j] = 0.5  # Split pot
                else:
                    # Approximate: higher bucket = stronger hand
                    equity = 0.5 + 0.3 * (i - j) / self.num_hands
                    self.call_matrix[i][j] = np.clip(equity, 0.0, 1.0)

    def _get_card_rank(self, card) -> int:
        """
        Get rank of card.
        
        Args:
            card: Card object or integer
            
        Returns:
            Rank (0-2 for Leduc: J, Q, K)
        """
        if isinstance(card, int):
            return card // 2
        elif hasattr(card, 'rank'):
            rank_map = {'J': 0, 'jack': 0, 'Q': 1, 'queen': 1, 'K': 2, 'king': 2}
            return rank_map.get(str(card.rank).lower(), 0)
        return 0

    def call_value(self, ranges: np.ndarray, result: np.ndarray):
        """
        Compute counterfactual values for call nodes (showdown).
        
        Args:
            ranges: Range vectors for both players [player1_range, player2_range]
            result: Output array to store CFVs for both players [cfv1, cfv2]
        """
        if self.call_matrix is None:
            self._compute_equity_matrices()
        
        player1_range = ranges[0]
        player2_range = ranges[1]
        
        # Player 1 CFV: sum over opponent hands of equity * opponent prob
        result[0] = np.dot(self.call_matrix, player2_range)
        
        # Player 2 CFV: sum over opponent hands (player 1) of (1 - equity) * opp prob
        # Note: call_matrix[i][j] is equity for hand i vs hand j from player 1's perspective
        result[1] = np.dot(1.0 - self.call_matrix.T, player1_range)

    def fold_value(self, ranges: np.ndarray, result: np.ndarray, folding_player: int):
        """
        Compute counterfactual values for fold nodes.
        
        Args:
            ranges: Range vectors for both players
            result: Output array to store CFVs for both players
            folding_player: Which player folded (0 or 1)
        """
        # When a player folds, opponent wins the pot
        # Folding player gets -pot, opponent gets +pot
        # Simplified: uniform payoff
        if folding_player == 0:
            result[0] = -1.0 * np.ones(len(ranges[0]))
            result[1] = 1.0 * np.ones(len(ranges[1]))
        else:
            result[0] = 1.0 * np.ones(len(ranges[0]))
            result[1] = -1.0 * np.ones(len(ranges[1]))

    def get_call_matrix(self) -> np.ndarray:
        """
        Get equity matrix for showdowns.
        
        Returns:
            Matrix where [i][j] = equity of hand i vs hand j
        """
        if self.call_matrix is None:
            self._compute_equity_matrices()
        return self.call_matrix

    def tree_node_call_value(self, ranges: np.ndarray, result: np.ndarray):
        """
        Compute CFVs at call node for both players.
        
        Args:
            ranges: Range vectors [player1, player2]
            result: Output CFVs [cfv1, cfv2]
        """
        self.call_value(ranges, result)

    def tree_node_fold_value(self, ranges: np.ndarray, result: np.ndarray, folding_player: int):
        """
        Compute CFVs at fold node for both players.
        
        Args:
            ranges: Range vectors [player1, player2]
            result: Output CFVs [cfv1, cfv2]
            folding_player: Player who folded (0 or 1)
        """
        self.fold_value(ranges, result, folding_player)
    
    def compute_expected_value(self, player_range: np.ndarray, 
                               opponent_range: np.ndarray,
                               pot_size: float = 100.0) -> np.ndarray:
        """
        Compute expected value for player range vs opponent range.
        
        Args:
            player_range: Player's range (probability distribution)
            opponent_range: Opponent's range (probability distribution)
            pot_size: Pot size
            
        Returns:
            Expected value for each hand in player's range
        """
        if self.call_matrix is None:
            self._compute_equity_matrices()
        
        ev = np.dot(self.call_matrix, opponent_range) * pot_size
        return ev
