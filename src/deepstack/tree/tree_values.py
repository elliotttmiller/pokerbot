"""
Tree values utilities for DeepStack.
Ported from DeepStack Lua tree_values.lua.
"""
import numpy as np

def compute_tree_values(node):
    """
    Recursively computes expected value and best response value for strategy profile on public tree.
    Args:
        node: root node of the tree (dict with children, strategy, terminal)
    """
    if node.get('terminal', False):
        node['cf_values'] = np.zeros(2)
        node['cf_values_br'] = np.zeros(2)
        return
    # Placeholder: propagate values from children
    for child in node['children']:
        compute_tree_values(child)
    node['cf_values'] = np.mean([child['cf_values'] for child in node['children']], axis=0)
    node['cf_values_br'] = np.mean([child['cf_values_br'] for child in node['children']], axis=0)
