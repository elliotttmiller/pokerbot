"""
TerminalEquity: DeepStack-style terminal equity evaluation for showdown and fold nodes.
Implements call_value, fold_value, get_call_matrix, tree_node_call_value, tree_node_fold_value.
"""
import numpy as np

class TerminalEquity:
    def __init__(self):
        self.board = None
        self.call_matrix = None

    def set_board(self, board):
        self.board = board
        # TODO: Build call matrix for showdown equity
        self.call_matrix = np.ones((13, 13)) / 2  # Placeholder for equity matrix

    def call_value(self, ranges, result):
        # TODO: Compute CFVs for showdown (no fold)
        result[:] = np.dot(ranges, self.call_matrix)

    def fold_value(self, ranges, result):
        # TODO: Compute CFVs for fold nodes
        result[:] = -np.dot(ranges, self.call_matrix)

    def get_call_matrix(self):
        return self.call_matrix

    def tree_node_call_value(self, ranges, result):
        # TODO: Compute CFVs for both players at showdown
        result[:] = np.dot(ranges, self.call_matrix)

    def tree_node_fold_value(self, ranges, result, folding_player):
        # TODO: Compute CFVs for both players when someone folds
        result[:] = -np.dot(ranges, self.call_matrix)
