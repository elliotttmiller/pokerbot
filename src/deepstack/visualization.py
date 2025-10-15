"""Visualization helpers for DeepStack trees and strategies."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np


def collect_tree_stats(root) -> Dict[str, Any]:
    """Traverse tree and collect simple stats: node counts, depth, branching."""
    stats = {"nodes": 0, "terminals": 0, "max_depth": 0, "branching": []}

    def dfs(node, depth=0):
        stats["nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)
        children = getattr(node, "children", []) or []
        stats["branching"].append(len(children))
        if not children:
            stats["terminals"] += 1
        for c in children:
            dfs(c, depth + 1)

    dfs(root)
    stats["avg_branching"] = float(np.mean(stats["branching"])) if stats["branching"] else 0.0
    return stats


def export_strategy_table(strategy: Dict[str, np.ndarray]) -> Dict[str, list]:
    """Convert strategy dict to JSON-serializable structure."""
    return {k: v.tolist() for k, v in strategy.items()}


def _entropy(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log2(probs)).sum())


def summarize_cfr_results(results: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """Summarize CFR outputs with regret statistics and entropy diagnostics."""

    regrets = results.get("regrets", {}) or {}
    strategy = results.get("strategy", {}) or {}

    total_positive_regret = float(sum(np.maximum(reg, 0).sum() for reg in regrets.values()))
    max_regret = float(max((np.max(np.abs(reg)) for reg in regrets.values()), default=0.0))
    avg_regret = float(np.mean([np.mean(np.abs(reg)) for reg in regrets.values()])) if regrets else 0.0

    top_nodes = []
    if regrets:
        regret_items = [
            (node, float(np.maximum(reg, 0).sum())) for node, reg in regrets.items()
        ]
        regret_items.sort(key=lambda item: item[1], reverse=True)
        top_nodes = regret_items[:top_k]

    entropies = {node: _entropy(np.asarray(prob)) for node, prob in strategy.items()}
    entropy_stats = {
        "mean_entropy": float(np.mean(list(entropies.values()))) if entropies else 0.0,
        "max_entropy": float(max(entropies.values())) if entropies else 0.0,
        "min_entropy": float(min(entropies.values())) if entropies else 0.0,
    }

    return {
        "aggregate": {
            "total_positive_regret": total_positive_regret,
            "max_regret": max_regret,
            "avg_regret": avg_regret,
            "strategy_count": len(strategy),
        },
        "top_regret_nodes": [{"node": node, "positive_regret": value} for node, value in top_nodes],
        "entropy": entropy_stats,
    }
