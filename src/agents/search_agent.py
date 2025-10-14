"""Real-time search agent with blueprint strategy.

Implements two-stage approach from Pluribus:
1. Blueprint strategy: Pre-computed via CFR (offline)
2. Real-time search: Depth-limited search during critical decisions (online)
"""

from typing import List, Optional, Tuple
import numpy as np

from ..game import Action, Card, GameState
from .champion_agent import ChampionAgent


class SearchAgent(ChampionAgent):
    """
    Champion Agent enhanced with real-time search capabilities.
    
    Uses blueprint strategy for most decisions, but employs depth-limited
    search for critical situations (large pots, close decisions).
    """
    
    def __init__(self,
                 name: str = "SearchAgent",
                 search_depth: int = 2,
                 search_threshold_pot: int = 200,
                 use_pretrained: bool = True,
                 **kwargs):
        """
        Initialize search agent.
        
        Args:
            name: Agent name
            search_depth: Maximum depth for real-time search
            search_threshold_pot: Minimum pot size to trigger search
            use_pretrained: Load pre-trained models
            **kwargs: Additional args for ChampionAgent
        """
        super().__init__(name=name, use_pretrained=use_pretrained, **kwargs)
        self.search_depth = search_depth
        self.search_threshold_pot = search_threshold_pot
    
    def choose_action(self,
                     hole_cards: List[Card],
                     community_cards: List[Card],
                     pot: int,
                     current_bet: int,
                     player_stack: int,
                     opponent_bet: int) -> Tuple[Action, int]:
        """
        Choose action using blueprint or real-time search.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        # Determine if we should use real-time search
        use_search = self._should_use_search(
            pot, current_bet, community_cards
        )
        
        if use_search:
            # Use depth-limited search for critical decisions
            return self._depth_limited_search(
                hole_cards, community_cards, pot,
                current_bet, player_stack, opponent_bet
            )
        else:
            # Use blueprint strategy (Champion Agent's ensemble)
            return super().choose_action(
                hole_cards, community_cards, pot,
                current_bet, player_stack, opponent_bet
            )
    
    def _should_use_search(self,
                          pot: int,
                          current_bet: int,
                          community_cards: List[Card]) -> bool:
        """
        Determine if real-time search should be used.
        
        Args:
            pot: Current pot size
            current_bet: Current bet
            community_cards: Community cards
        
        Returns:
            True if search should be used
        """
        # Use search for large pots
        if pot >= self.search_threshold_pot:
            return True
        
        # Use search on turn and river (more critical decisions)
        if len(community_cards) >= 4:
            return True
        
        # Use search for large bets
        if current_bet > pot * 0.5:
            return True
        
        return False
    
    def _depth_limited_search(self,
                             hole_cards: List[Card],
                             community_cards: List[Card],
                             pot: int,
                             current_bet: int,
                             player_stack: int,
                             opponent_bet: int) -> Tuple[Action, int]:
        """
        Perform depth-limited search to find best action.
        
        Uses minimax with alpha-beta pruning.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet to call
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            Tuple of (Action, raise_amount)
        """
        # Create game state for search
        state = self._create_search_state(
            hole_cards, community_cards, pot,
            current_bet, player_stack, opponent_bet
        )
        
        # Get available actions
        available_actions = self._get_search_actions(
            current_bet, player_stack, pot
        )
        
        # Evaluate each action
        best_action = None
        best_value = float('-inf')
        best_raise_amount = 0
        
        for action_type, raise_amount in available_actions:
            # Simulate action
            value = self._evaluate_action(
                state, action_type, raise_amount,
                depth=0, max_depth=self.search_depth,
                alpha=float('-inf'), beta=float('inf'),
                is_maximizing=False  # Opponent's turn next
            )
            
            if value > best_value:
                best_value = value
                best_action = action_type
                best_raise_amount = raise_amount
        
        # Fallback to blueprint if search fails
        if best_action is None:
            return super().choose_action(
                hole_cards, community_cards, pot,
                current_bet, player_stack, opponent_bet
            )
        
        return best_action, best_raise_amount
    
    def _get_search_actions(self,
                           current_bet: int,
                           player_stack: int,
                           pot: int) -> List[Tuple[Action, int]]:
        """
        Get available actions for search.
        
        Args:
            current_bet: Current bet to call
            player_stack: Player's remaining stack
            pot: Current pot size
        
        Returns:
            List of (action, raise_amount) tuples
        """
        actions = []
        
        # Fold
        actions.append((Action.FOLD, 0))
        
        # Call or Check
        if current_bet == 0:
            actions.append((Action.CHECK, 0))
        else:
            actions.append((Action.CALL, 0))
        
        # Raises (discretized)
        if player_stack > 0:
            # Small raise (0.5x pot)
            small_raise = min(int(pot * 0.5), player_stack)
            if small_raise >= 20:
                actions.append((Action.RAISE, small_raise))
            
            # Medium raise (1x pot)
            med_raise = min(pot, player_stack)
            if med_raise >= 20 and med_raise != small_raise:
                actions.append((Action.RAISE, med_raise))
            
            # Large raise (2x pot)
            large_raise = min(int(pot * 2), player_stack)
            if large_raise >= 20 and large_raise != med_raise:
                actions.append((Action.RAISE, large_raise))
        
        return actions
    
    def _evaluate_action(self,
                        state: dict,
                        action: Action,
                        raise_amount: int,
                        depth: int,
                        max_depth: int,
                        alpha: float,
                        beta: float,
                        is_maximizing: bool) -> float:
        """
        Evaluate action using minimax with alpha-beta pruning.
        
        Args:
            state: Current game state
            action: Action to evaluate
            raise_amount: Raise amount if applicable
            depth: Current search depth
            max_depth: Maximum search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: True if maximizing player's turn
        
        Returns:
            Estimated value of action
        """
        # Terminal conditions
        if depth >= max_depth:
            return self._evaluate_state(state)
        
        if action == Action.FOLD:
            return -state['pot'] * 0.5  # Lost current investment
        
        # Simulate action effect on state
        new_state = self._apply_search_action(state, action, raise_amount)
        
        # Recursive evaluation
        if is_maximizing:
            max_eval = float('-inf')
            for next_action, next_raise in self._get_search_actions(
                new_state['current_bet'],
                new_state['player_stack'],
                new_state['pot']
            ):
                eval_score = self._evaluate_action(
                    new_state, next_action, next_raise,
                    depth + 1, max_depth, alpha, beta, False
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            for next_action, next_raise in self._get_search_actions(
                new_state['current_bet'],
                new_state['player_stack'],
                new_state['pot']
            ):
                eval_score = self._evaluate_action(
                    new_state, next_action, next_raise,
                    depth + 1, max_depth, alpha, beta, True
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
    
    def _evaluate_state(self, state: dict) -> float:
        """
        Evaluate state value (heuristic for leaf nodes).
        
        Args:
            state: Game state dictionary
        
        Returns:
            Estimated value
        """
        # Use blueprint strategy to estimate value
        # This is a simplified heuristic
        pot_odds = state['current_bet'] / (state['pot'] + state['current_bet']) if state['pot'] > 0 else 0
        hand_strength = 0.5  # Would use actual hand evaluation
        
        # Expected value
        ev = hand_strength * state['pot'] - (1 - hand_strength) * state['current_bet']
        
        return ev
    
    def _create_search_state(self,
                            hole_cards: List[Card],
                            community_cards: List[Card],
                            pot: int,
                            current_bet: int,
                            player_stack: int,
                            opponent_bet: int) -> dict:
        """
        Create state dictionary for search.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            pot: Current pot
            current_bet: Current bet
            player_stack: Player's stack
            opponent_bet: Opponent's bet
        
        Returns:
            State dictionary
        """
        return {
            'hole_cards': hole_cards,
            'community_cards': community_cards,
            'pot': pot,
            'current_bet': current_bet,
            'player_stack': player_stack,
            'opponent_bet': opponent_bet
        }
    
    def _apply_search_action(self,
                            state: dict,
                            action: Action,
                            raise_amount: int) -> dict:
        """
        Apply action to state (for search simulation).
        
        Args:
            state: Current state
            action: Action to apply
            raise_amount: Raise amount
        
        Returns:
            New state after action
        """
        new_state = state.copy()
        
        if action == Action.CALL:
            new_state['pot'] += state['current_bet']
            new_state['player_stack'] -= state['current_bet']
            new_state['current_bet'] = 0
        elif action == Action.RAISE:
            new_state['pot'] += raise_amount
            new_state['player_stack'] -= raise_amount
            new_state['current_bet'] = raise_amount
        elif action == Action.CHECK:
            pass  # No change
        
        return new_state
