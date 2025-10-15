#!/usr/bin/env python3
"""Streamlit dashboard to visualize strategy and tree stats.

Usage:
  streamlit run scripts/visualize_strategy.py
"""
import json
import os
import streamlit as st
import numpy as np

from src.deepstack.tree_builder import PokerTreeBuilder
from src.deepstack.tree_cfr import TreeCFR
from src.deepstack.visualization import (
    collect_tree_stats,
    export_strategy_table,
    summarize_cfr_results,
)

st.set_page_config(page_title="Pokerbot Strategy Dashboard", layout="wide")

st.title("Pokerbot Strategy & Tree Dashboard")

with st.sidebar:
    st.header("Tree Parameters")
    variant = st.selectbox("Game Variant", ["leduc", "holdem"], index=0)
    street = st.number_input("Start Street", min_value=0, max_value=3, value=0)
    p1_bet = st.number_input("P1 Bet", min_value=0, value=20)
    p2_bet = st.number_input("P2 Bet", min_value=0, value=20)
    current_player = st.selectbox("Current Player", [1, 2], index=0)
    bet_sizing = st.multiselect("Bet Sizing (x pot)", [0.5, 1.0, 2.0], default=[1.0])

    st.header("CFR")
    iters = st.number_input("Iterations", min_value=10, max_value=5000, value=200, step=10)

if st.button("Build & Solve"):
    builder = PokerTreeBuilder(game_variant=variant)
    root = builder.build_tree({
        "street": int(street),
        "bets": [int(p1_bet), int(p2_bet)],
        "current_player": int(current_player),
        "bet_sizing": bet_sizing or [1.0],
    })

    stats = collect_tree_stats(root)
    st.subheader("Tree Stats")
    st.json(stats)

    cfr = TreeCFR()
    res = cfr.run_cfr(root, iter_count=int(iters))

    summary = summarize_cfr_results(res)
    st.subheader("CFR Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total +Regret", f"{summary['aggregate']['total_positive_regret']:.2f}")
    col2.metric("Max Regret", f"{summary['aggregate']['max_regret']:.2f}")
    col3.metric("Avg Regret", f"{summary['aggregate']['avg_regret']:.2f}")

    entropy = summary["entropy"]
    st.caption(
        f"Entropy → mean: {entropy['mean_entropy']:.3f}, "
        f"min: {entropy['min_entropy']:.3f}, max: {entropy['max_entropy']:.3f}"
    )

    if summary["top_regret_nodes"]:
        st.subheader("Top Nodes by Positive Regret")
        st.dataframe(summary["top_regret_nodes"])

    st.subheader("Strategy (sample)")
    sample = dict(list(res["strategy"].items())[:10])
    st.json(export_strategy_table(sample))

    st.success("Done")
else:
    st.info("Set parameters and click 'Build & Solve'")
