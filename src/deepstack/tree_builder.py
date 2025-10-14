"""
PokerTreeBuilder: DeepStack-style public game tree construction for poker variants.

Builds complete game trees with proper node types, betting rounds, board cards,
pot sizes, and action children. Supports configurable bet sizing abstractions.

Based on the original DeepStack TreeBuilder module.
"""
from typing import List, Dict, Optional, Tuple
import copy

# DeepStack constants
PLAYERS = {'chance': 0, 'P1': 1, 'P2': 2}
NODE_TYPES = {'terminal_fold': -2, 'terminal_call': -1, 'chance_node': 0, 'inner_node': 2}
ACTIONS = {'fold': -2, 'ccall': -1, 'raise': 1}
ACPC_ACTIONS = {'fold': 'fold', 'ccall': 'ccall', 'raise': 'raise'}
PLAYERS_COUNT = 2
STREETS_COUNT = 2  # Leduc Hold'em default


class PokerTreeNode:
    """
    Represents a node in the poker game tree.
    
    Attributes:
        node_type: Type of node (terminal, chance, inner)
        street: Betting round (0 for preflop, 1 for flop, etc.)
        board: Board cards dealt so far
        current_player: Player to act (0 for chance, 1 for P1, 2 for P2)
        bets: Array of total chips committed by each player [p1_bet, p2_bet]
        pot: Half the pot size (min of bets)
        children: List of child nodes
        actions: List of actions leading to children
        parent: Parent node (optional)
    """
    
    def __init__(self, node_type: str, street: int, board: List, 
                 current_player: int, bets: List[int], pot: int, 
                 children: Optional[List] = None, actions: Optional[List] = None,
                 parent: Optional['PokerTreeNode'] = None):
        self.node_type = node_type
        self.street = street
        self.board = board if board else []
        self.current_player = current_player
        self.bets = bets if isinstance(bets, list) else [bets, bets]
        self.pot = pot
        self.children = children if children else []
        self.actions = actions if actions else []
        self.parent = parent
        self.board_string = ''.join(str(c) for c in board) if board else ''
        
        # Strategy storage (filled by CFR)
        self.strategy = None
        self.average_strategy = None
        self.strategy_weight = 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node."""
        return self.node_type in ['terminal_fold', 'terminal_call']
    
    def is_chance(self) -> bool:
        """Check if this is a chance node."""
        return self.node_type == 'chance_node'
    
    def add_child(self, child: 'PokerTreeNode', action: str):
        """Add a child node with corresponding action."""
        self.children.append(child)
        self.actions.append(action)
        child.parent = self


class PokerTreeBuilder:
    """
    Builds public game trees for poker variants.
    
    Supports:
    - Multiple betting rounds (streets)
    - Configurable bet sizing
    - Proper terminal node handling
    - Chance nodes for card dealing
    """
    
    def __init__(self, game_variant: str = 'leduc', 
                 stack_size: int = 1000,
                 small_blind: int = 10,
                 big_blind: int = 20):
        """
        Initialize tree builder.
        
        Args:
            game_variant: 'leduc' or 'holdem'
            stack_size: Starting stack size
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.game_variant = game_variant
        self.stack_size = stack_size
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Betting round names
        if game_variant == 'leduc':
            self.street_names = ['preflop', 'flop']
            self.num_streets = 2
        else:
            self.street_names = ['preflop', 'flop', 'turn', 'river']
            self.num_streets = 4

    def build_tree(self, params: Dict) -> PokerTreeNode:
        """
        Build the public game tree from root node parameters.
        
        Args:
            params: Dictionary with keys:
                - street: Starting betting round (0-indexed)
                - bets: List of bets [player1_bet, player2_bet]
                - current_player: Player to act (1 or 2)
                - board: List of board cards
                - limit_to_street: If True, don't build beyond current street
                - bet_sizing: List of bet multipliers (e.g., [1.0] for pot bet)
                
        Returns:
            Root node of the constructed tree
        """
        # Extract parameters with defaults
        street = params.get('street', 0)
        bets = params.get('bets', [self.big_blind, self.big_blind])
        current_player = params.get('current_player', PLAYERS['P1'])
        board = params.get('board', [])
        limit_to_street = params.get('limit_to_street', False)
        bet_sizing = params.get('bet_sizing', [1.0])  # Default: pot-sized bets
        
        # Convert bets to list if needed
        if not isinstance(bets, list):
            bets = [bets, bets]
        
        pot = min(bets)
        
        # Create root node
        root = PokerTreeNode(
            node_type='inner_node',
            street=street,
            board=board,
            current_player=current_player,
            bets=bets,
            pot=pot
        )
        
        # Build tree recursively
        self._build_subtree(root, bet_sizing, limit_to_street)
        
        return root
    
    def _build_subtree(self, node: PokerTreeNode, bet_sizing: List[float],
                       limit_to_street: bool):
        """
        Recursively build subtree from node.
        
        Args:
            node: Current node to expand
            bet_sizing: Bet size multipliers
            limit_to_street: Whether to limit to current street
        """
        # Terminal node - no children
        if node.is_terminal():
            return
        
        # Chance node - add children for possible board cards
        if node.is_chance():
            self._build_chance_children(node, bet_sizing, limit_to_street)
            return
        
        # Regular action node - add fold, call, raise children
        self._build_action_children(node, bet_sizing, limit_to_street)
    
    def _build_action_children(self, node: PokerTreeNode, bet_sizing: List[float],
                                limit_to_street: bool):
        """
        Build children for action node (fold, call, raise).
        
        Args:
            node: Current node
            bet_sizing: Bet size multipliers
            limit_to_street: Limit to current street
        """
        current_player = node.current_player
        opponent = 3 - current_player  # Switch between 1 and 2
        
        player_bet = node.bets[current_player - 1]
        opponent_bet = node.bets[opponent - 1]
        
        # Player can fold if opponent has raised
        if player_bet < opponent_bet:
            fold_node = PokerTreeNode(
                node_type='terminal_fold',
                street=node.street,
                board=node.board,
                current_player=current_player,
                bets=node.bets.copy(),
                pot=node.pot,
                parent=node
            )
            node.add_child(fold_node, 'fold')
        
        # Player can call/check
        new_bets = node.bets.copy()
        new_bets[current_player - 1] = opponent_bet  # Match opponent's bet
        
        # Check if this completes the street
        if player_bet == opponent_bet:
            # Check action
            action_name = 'check'
        else:
            # Call action
            action_name = 'call'
        
        # Determine if street ends after call
        if new_bets[0] == new_bets[1]:
            # Both players matched - street ends
            if node.street < self.num_streets - 1 and not limit_to_street:
                # Move to next street via chance node
                chance_node = PokerTreeNode(
                    node_type='chance_node',
                    street=node.street + 1,
                    board=node.board.copy(),
                    current_player=PLAYERS['chance'],
                    bets=new_bets,
                    pot=min(new_bets),
                    parent=node
                )
                node.add_child(chance_node, action_name)
                self._build_chance_children(chance_node, bet_sizing, limit_to_street)
            else:
                # Final street - showdown
                terminal_node = PokerTreeNode(
                    node_type='terminal_call',
                    street=node.street,
                    board=node.board,
                    current_player=current_player,
                    bets=new_bets,
                    pot=min(new_bets),
                    parent=node
                )
                node.add_child(terminal_node, action_name)
        else:
            # Call but not matched yet - opponent acts next
            call_node = PokerTreeNode(
                node_type='inner_node',
                street=node.street,
                board=node.board,
                current_player=opponent,
                bets=new_bets,
                pot=min(new_bets),
                parent=node
            )
            node.add_child(call_node, action_name)
            self._build_subtree(call_node, bet_sizing, limit_to_street)
        
        # Player can raise (if not all-in)
        for multiplier in bet_sizing:
            raise_amount = int(node.pot * multiplier)
            new_bet = opponent_bet + raise_amount
            
            # Check if within stack limits
            if new_bet <= self.stack_size:
                raise_bets = node.bets.copy()
                raise_bets[current_player - 1] = new_bet
                
                raise_node = PokerTreeNode(
                    node_type='inner_node',
                    street=node.street,
                    board=node.board,
                    current_player=opponent,
                    bets=raise_bets,
                    pot=min(raise_bets),
                    parent=node
                )
                node.add_child(raise_node, f'raise_{multiplier}')
                self._build_subtree(raise_node, bet_sizing, limit_to_street)
    
    def _build_chance_children(self, node: PokerTreeNode, bet_sizing: List[float],
                                limit_to_street: bool):
        """
        Build children for chance node (deal board cards).
        
        Args:
            node: Chance node
            bet_sizing: Bet size multipliers
            limit_to_street: Limit to current street
        """
        # For simplicity, create a single child with a generic board card
        # In production, would create child for each possible board card
        
        new_board = node.board.copy()
        new_board.append(f'board_{node.street}')  # Placeholder board card
        
        # First player acts on new street
        action_node = PokerTreeNode(
            node_type='inner_node',
            street=node.street,
            board=new_board,
            current_player=PLAYERS['P1'],
            bets=node.bets.copy(),
            pot=node.pot,
            parent=node
        )
        
        node.add_child(action_node, 'deal')
        self._build_subtree(action_node, bet_sizing, limit_to_street)
