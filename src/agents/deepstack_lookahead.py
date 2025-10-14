"""
DeepStack Lookahead module integration for pokerbot agents.
Wraps key methods: build_lookahead, resolve_first_node, resolve, get_chance_action_cfv, get_results.
"""
import numpy as np
from src.deepstack.tree_builder import PokerTreeBuilder, PLAYERS, NODE_TYPES, ACTIONS, ACPC_ACTIONS
from src.deepstack.terminal_equity import TerminalEquity
from src.deepstack.tree_cfr import TreeCFR
from src.deepstack.cfrd_gadget import CFRDGadget

class DeepStackLookahead:
    def __init__(self, params=None):
        self.params = params or {}
        self.tree = None
        self.last_results = None

    def build_lookahead(self, player, opponent, board, **kwargs):
        builder = PokerTreeBuilder()
        params = {
            'street': kwargs.get('street', 0),
            'bets': kwargs.get('bets', [0, 0]),
            'current_player': player,
            'board': board,
            'limit_to_street': kwargs.get('limit_to_street', False),
            'bet_sizing': kwargs.get('bet_sizing', None)
        }
        self.tree = builder.build_tree(params)

    def resolve_first_node(self, player_range, opponent_range):
        cfr = TreeCFR()
        strategy = cfr.run_cfr(self.tree, [player_range, opponent_range], 1000)
        terminal = TerminalEquity()
        terminal.set_board(self.tree.board)
        achieved_cfvs = np.dot(strategy, opponent_range)
        children_cfvs = np.array([np.dot(strategy[i], opponent_range) for i in range(strategy.shape[0])])
        self.last_results = {
            'strategy': strategy,
            'achieved_cfvs': achieved_cfvs,
            'children_cfvs': children_cfvs
        }

    def resolve(self, player_range, opponent_cfvs):
        cfr = TreeCFR()
        strategy = cfr.run_cfr(self.tree, [player_range, opponent_cfvs], 1000)
        achieved_cfvs = opponent_cfvs
        children_cfvs = np.array([np.dot(strategy[i], opponent_cfvs) for i in range(strategy.shape[0])])
        self.last_results = {
            'strategy': strategy,
            'achieved_cfvs': achieved_cfvs,
            'children_cfvs': children_cfvs
        }

    def get_chance_action_cfv(self, action_index, board):
        gadget = CFRDGadget(board, None, None)
        cfvs = gadget.compute_opponent_range(np.ones(10), 0)
        return cfvs

    def get_results(self):
        """
        Returns results of re-solving: strategy, achieved_cfvs, children_cfvs.
        Returns:
            dict
        """
        return self.last_results or {}
