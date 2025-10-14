"""Evaluation framework for comparing poker agents."""

from typing import Dict, List

from ..agents import BaseAgent
from ..game import Action, GameState


class Evaluator:
    """Evaluates and compares poker agents."""
    
    def __init__(self, agents: List[BaseAgent]):
        """
        Initialize evaluator.
        
        Args:
            agents: List of agents to evaluate
        """
        self.agents = agents
    
    def evaluate_agents(self, num_hands: int = 1000, verbose: bool = True) -> Dict:
        """
        Evaluate agents against each other.
        
        Args:
            num_hands: Number of hands to play
            verbose: Print progress
        
        Returns:
            Dictionary with evaluation results
        """
        results = {
            'wins': {agent.name: 0 for agent in self.agents},
            'losses': {agent.name: 0 for agent in self.agents},
            'ties': 0,
            'total_winnings': {agent.name: 0 for agent in self.agents},
            'hand_results': []
        }
        
        for hand_num in range(num_hands):
            # Play one hand
            hand_result = self._play_evaluation_hand()
            
            # Update results
            if hand_result['winner'] is not None:
                winner_name = hand_result['winner']
                results['wins'][winner_name] += 1
                results['total_winnings'][winner_name] += hand_result['pot']
                
                for agent in self.agents:
                    if agent.name != winner_name:
                        results['losses'][agent.name] += 1
                        results['total_winnings'][agent.name] -= hand_result['pot'] // (len(self.agents) - 1)
            else:
                results['ties'] += 1
            
            results['hand_results'].append(hand_result)
            
            # Print progress
            if verbose and (hand_num + 1) % 100 == 0:
                print(f"Completed {hand_num + 1}/{num_hands} hands")
                self._print_results(results, hand_num + 1)
        
        if verbose:
            print("\n" + "="*60)
            print(f"Final Results after {num_hands} hands:")
            print("="*60)
            self._print_results(results, num_hands)
        
        return results
    
    def _play_evaluation_hand(self) -> Dict:
        """
        Play one evaluation hand between agents.
        
        Returns:
            Dictionary with hand result
        """
        game = GameState(num_players=len(self.agents))
        game.reset()
        
        done = False
        current_agent_idx = 0
        
        while not done:
            player = game.players[current_agent_idx]
            agent = self.agents[current_agent_idx]
            
            if not player.folded and not player.all_in:
                # Get action from agent
                action, raise_amount = agent.choose_action(
                    hole_cards=player.hand,
                    community_cards=game.community_cards,
                    pot=game.pot,
                    current_bet=game.current_bet - player.current_bet,
                    player_stack=player.stack,
                    opponent_bet=game.current_bet
                )
                
                # Apply action
                game.apply_action(current_agent_idx, action, raise_amount)
            
            # Move to next player
            current_agent_idx = (current_agent_idx + 1) % len(self.agents)
            
            # Check if betting round is complete
            if self._is_betting_complete(game):
                if game.is_hand_complete():
                    done = True
                else:
                    game.advance_betting_round()
                    current_agent_idx = 0
        
        # Determine winner
        winners = game.get_winners()
        
        result = {
            'pot': game.pot,
            'winner': self.agents[winners[0]].name if len(winners) == 1 else None,
            'winners': [self.agents[idx].name for idx in winners],
            'community_cards': [str(card) for card in game.community_cards]
        }
        
        return result
    
    def _is_betting_complete(self, game: GameState) -> bool:
        """Check if current betting round is complete."""
        active_players = [p for p in game.players if not p.folded and not p.all_in]
        
        if len(active_players) <= 1:
            return True
        
        # Check if all active players have equal bets
        bets = [p.current_bet for p in active_players]
        return len(set(bets)) == 1
    
    def _print_results(self, results: Dict, num_hands: int):
        """Print evaluation results."""
        print("\nAgent Performance:")
        print("-" * 60)
        
        for agent in self.agents:
            name = agent.name
            wins = results['wins'][name]
            losses = results['losses'][name]
            win_rate = wins / num_hands * 100
            total_winnings = results['total_winnings'][name]
            
            print(f"{name:20s} - Wins: {wins:4d} ({win_rate:5.2f}%) - "
                  f"Losses: {losses:4d} - Winnings: ${total_winnings:+8d}")
        
        ties = results['ties']
        tie_rate = ties / num_hands * 100
        print(f"{'Ties':20s} - {ties:4d} ({tie_rate:5.2f}%)")
    
    def head_to_head(self, agent1_idx: int, agent2_idx: int, 
                     num_hands: int = 1000) -> Dict:
        """
        Run head-to-head evaluation between two agents.
        
        Args:
            agent1_idx: Index of first agent
            agent2_idx: Index of second agent
            num_hands: Number of hands to play
        
        Returns:
            Dictionary with head-to-head results
        """
        agents = [self.agents[agent1_idx], self.agents[agent2_idx]]
        temp_evaluator = Evaluator(agents)
        return temp_evaluator.evaluate_agents(num_hands, verbose=True)
