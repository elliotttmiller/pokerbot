# Pokerbot Optimization & Feature Implementation TODO

Status: planned and initial scaffolding implemented. Items are prioritized by impact vs. change size.

- [x] Automated Hyperparameter Tuning (Optuna)
  - [x] Add Optuna dependency
  - [x] Implement tuning utilities: `src/evaluation/tuning.py`
  - [x] Provide CLI: `scripts/tune_hyperparams.py`
  - [x] Extend to ChampionAgent parameters (CFR/DQN/equity weights)

- [x] Model Pruning & Quantization (TensorFlow / Keras)
  - [x] Add tensorflow-model-optimization dependency
  - [x] Implement utilities: `src/utils/model_optimization.py`
  - [x] Provide CLI: `scripts/optimize_model.py`
  - [x] Integrate into training pipeline post-save hook

- [x] Strategy Visualization Dashboard
  - [x] Implement visualization helpers: `src/deepstack/visualization.py`
  - [x] Provide Streamlit app: `scripts/visualize_strategy.py`
  - [x] Enrich with exploitability tracking and per-node stats

- [x] Automated Data Validation & Integrity Checks
  - [x] Implement validators: `src/utils/data_validation.py`
  - [x] Provide CLI: `scripts/validate_data.py`
  - [x] Add pre-train validation step in training scripts

- [x] Meta-Learning & Opponent Modeling (scaffolding)
  - [x] Opponent model: `src/agents/opponent_model.py`
  - [x] Meta agent wrapper: `src/agents/meta_agent.py`
  - [x] Wire into training/evaluation loops with adaptive curriculum

Notes:
- All new scripts have `--help` and safe defaults for smoketest usage.
- No breaking changes to existing training scripts; integration is additive.
- Future work: integrate optimization hooks into `train_champion.py` and add evaluation-driven auto-promotion of optimized models.
