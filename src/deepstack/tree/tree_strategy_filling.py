"""
Tree strategy filling utilities for DeepStack.
Ported from DeepStack Lua tree_strategy_filling.lua.
"""
import numpy as np

def fill_strategy(node, board_count, card_count):
    """
    Recursively fills strategies for public tree nodes.
    Args:
        node: root node of the tree (dict with children, current_player, terminal)
        board_count: number of possible boards
        card_count: number of possible cards
    """
    if node.get('terminal', False):
        return
    if node['current_player'] == 'chance':
        assert len(node['children']) == board_count
        node['strategy'] = np.zeros((len(node['children']), card_count))
        for i, child in enumerate(node['children']):
            node['strategy'][i, :] = 1.0 / (board_count - 2)
    else:
        node['strategy'] = np.ones((len(node['children']), card_count)) / len(node['children'])
    for child in node['children']:
        fill_strategy(child, board_count, card_count)
