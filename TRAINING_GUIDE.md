# PokerBot Training Guide
## Complete System Training and Testing Manual

This guide provides comprehensive, step-by-step instructions for training your world-class PokerBot agent from scratch to championship-level performance.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Architecture Overview](#system-architecture-overview)
3. [Training Pipeline Stages](#training-pipeline-stages)
4. [Quick Start: Smoketest](#quick-start-smoketest)
5. [Standard Training](#standard-training)
6. [Production Training](#production-training)
7. [Model Evaluation](#model-evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps: Vision System](#next-steps-vision-system)

---

## Prerequisites

### 1. Environment Setup

```bash
# Clone repository (if not already done)
git clone https://github.com/elliotttmiller/pokerbot.git
cd pokerbot

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow, torch, numpy; print('✓ All dependencies installed')"
```

### 2. Data Preparation

The training pipeline requires:

**Training Data** (Located in `data/deepstacked_training/samples/train_samples/`):
- `train_inputs.pt` - Training input tensors
- `train_targets.pt` - Training target values
- `train_mask.pt` - Training masks
- `valid_inputs.pt` - Validation inputs
- `valid_targets.pt` - Validation targets
- `valid_mask.pt` - Validation masks

**Equity Tables** (Located in `data/equity_tables/`):
- `preflop_equity.json` - Preflop hand equity calculations

**Verify Data:**
```bash
python scripts/validate_data.py
```

Expected output:
```
✓ Training samples validated
✓ Equity table validated
✓ All data files present and valid
```

### 3. Directory Structure

The training pipeline will create:
```
pokerbot/
├── models/              # Trained models
│   ├── versions/       # Versioned models
│   ├── checkpoints/    # Training checkpoints
│   └── reports/        # Training reports
├── logs/               # Training logs
└── data/               # Training data (pre-existing)
```

---

## System Architecture Overview

### Unified PokerBot Agent

The `PokerBotAgent` combines multiple AI techniques:

**Components (All Configurable):**
1. **CFR/CFR+** - Game-theoretic optimal play
2. **DQN** - Deep Q-Network for pattern learning
3. **DeepStack** - Continual re-solving with lookahead
4. **Opponent Modeling** - Adaptive exploitation
5. **Pre-trained Models** - Championship-level knowledge

**Ensemble Decision Making:**
- Each component votes on actions
- Weighted voting system (configurable weights)
- Final decision optimized through ensemble

### Training Pipeline (3-Stage Progressive Training)

**Stage 1: CFR Warmup** 
- Trains game-theoretic foundation
- Builds baseline strategy
- No opponent interaction yet

**Stage 2: Self-Play**
- Agent plays against itself
- DQN learns patterns through experience replay
- Epsilon-greedy exploration

**Stage 3: Vicarious Learning**
- Agent plays against diverse opponents
- Learns to exploit different play styles
- Curriculum learning with progressive difficulty

**Final: Evaluation & Comparison**
- Evaluates against standard opponents
- Compares with previous best model
- Promotes if better

---

## Training Pipeline Stages

### Stage 0: Data Validation

**Purpose:** Ensure training data integrity

**What happens:**
- Validates DeepStack training samples
- Validates equity tables
- Checks file formats and data consistency

**Duration:** < 5 seconds

**Failure:** Training stops if data is invalid

### Stage 1: CFR Warmup

**Purpose:** Build game-theoretic foundation

**What happens:**
- Runs CFR (Counterfactual Regret Minimization) iterations
- Builds initial strategy through regret matching
- No neural network training yet

**Configuration:**
- Smoketest: 10 iterations
- Standard: 200 iterations
- Production: 5000 iterations

**Duration:**
- Smoketest: ~10 seconds
- Standard: ~2-3 minutes
- Production: ~30-45 minutes

**Output:** CFR strategy saved to checkpoints

### Stage 2: Self-Play

**Purpose:** Learn through experience

**What happens:**
- Agent plays against copy of itself
- Stores experiences in replay buffer
- Trains DQN network on batches of experiences
- Gradually reduces exploration (epsilon decay)

**Configuration:**
- Smoketest: 10 episodes
- Standard: 100 episodes
- Production: 2000 episodes

**Duration:**
- Smoketest: ~30 seconds
- Standard: ~5-10 minutes
- Production: ~2-4 hours

**Metrics Tracked:**
- Average reward
- Win rate
- Epsilon value (exploration rate)
- Memory buffer size

**Output:** Updated agent with learned patterns

### Stage 3: Vicarious Learning

**Purpose:** Learn to exploit different opponents

**What happens:**
- Agent plays against diverse opponents (Random, Fixed, CFR)
- Extra optimization steps (2x replay per episode)
- Curriculum learning (round-robin opponents)

**Configuration:**
- Smoketest: 10 episodes
- Standard: 100 episodes
- Production: 2000 episodes

**Duration:**
- Smoketest: ~30 seconds
- Standard: ~5-10 minutes
- Production: ~2-4 hours

**Opponents:**
- RandomAgent - Baseline random player
- FixedStrategyAgent - GTO-inspired fixed strategy
- CFRAgent - Pure CFR player

**Output:** Robust agent that exploits weaknesses

### Final Stage: Evaluation

**Purpose:** Assess final performance

**What happens:**
- Agent set to evaluation mode (no exploration)
- Plays validation hands against each opponent type
- Calculates win rates and average rewards
- Compares against previous best model
- Promotes if better (>50% win rate vs. previous best)

**Validation Hands:**
- Smoketest: 10 hands per opponent
- Standard: 50 hands per opponent
- Production: 500 hands per opponent

**Output:** 
- Performance metrics saved to logs
- Best model promoted if superior

---

## Quick Start: Smoketest

**Purpose:** Verify entire pipeline works (1-2 minutes total)

### Using PokerBot Agent (Recommended)

```bash
# Train with unified PokerBot agent
python scripts/train.py --agent-type pokerbot --mode smoketest --verbose

# With custom iterations
python scripts/train.py --agent-type pokerbot --mode smoketest --episodes 20 --verbose
```

### Using Legacy Champion Agent

```bash
# Train with legacy Champion agent (backward compatible)
python scripts/train.py --mode smoketest --verbose
```

### Expected Output:

```
[INFO] Starting championship training pipeline with pokerbot agent
[SAMPLES] Loaded train_inputs.pt: shape=(1000, 169), dtype=torch.float32, sum=84500.0
[SAMPLES] Loaded train_mask.pt: shape=(1000, 169), dtype=torch.float32, sum=169000.0
...
[INFO] Stage 0: Validating data...
✓ DeepStack samples validated
✓ Equity table validated

[INFO] Stage 1: CFR Warmup (10 iterations)...
[OK] CFR training complete

[INFO] Stage 2: Self-Play (10 episodes)...
[Stage2] Ep 5/10 | AvgR 12.50 | Win 60.00% | Eps 0.095 | Mem 50
[Stage2] Ep 10/10 | AvgR 15.30 | Win 70.00% | Eps 0.090 | Mem 100

[INFO] Stage 3: Vicarious Learning (10 episodes)...
[Stage3] Ep 5/10 | AvgR 18.20 | Win 60.00% | Eps 0.085 | Mem 150
[Stage3] Ep 10/10 | AvgR 20.50 | Win 65.00% | Eps 0.080 | Mem 200

[INFO] Final Evaluation...
  vs EvalRandom: Win Rate 90.00%, Avg Reward 45.50
  vs EvalFixed: Win Rate 70.00%, Avg Reward 25.30
  vs EvalCFR: Win Rate 55.00%, Avg Reward 12.50

[INFO] Comparing against previous best...
  Previous best does not exist
  Current model will be promoted to best

[INFO] Saved metrics to logs/training_metrics.json
[INFO] Champion training completed successfully
```

### Files Created:

```
models/
├── checkpoints/
│   └── champion_checkpoint.cfr
│   └── champion_checkpoint.keras
├── versions/
│   └── champion_current.cfr
│   └── champion_current.keras
│   └── champion_best.cfr  (promoted)
│   └── champion_best.keras  (promoted)
└── reports/
    └── (empty in smoketest)

logs/
└── training_metrics.json
```

---

## Standard Training

**Purpose:** Realistic training for strong agent (20-30 minutes total)

**When to use:** Development, testing improvements, baseline performance

### Command:

```bash
# Standard training with PokerBot
python scripts/train.py --agent-type pokerbot --mode standard --verbose

# With analysis report
python scripts/train.py --agent-type pokerbot --mode standard --verbose --report

# With custom seed for reproducibility
python scripts/train.py --agent-type pokerbot --mode standard --seed 12345 --verbose
```

### Configuration (Standard Mode):

```json
{
  "stage1_cfr_iterations": 200,
  "stage2_selfplay_episodes": 100,
  "stage3_vicarious_episodes": 100,
  "evaluation_interval": 10,
  "save_interval": 20,
  "batch_size": 8,
  "validation_hands": 50
}
```

### Timeline:

| Stage | Duration | Purpose |
|-------|----------|---------|
| Stage 1 (CFR) | ~2-3 min | Game theory foundation |
| Stage 2 (Self-play) | ~5-10 min | Pattern learning |
| Stage 3 (Vicarious) | ~5-10 min | Exploitation learning |
| Evaluation | ~2-5 min | Performance assessment |
| **Total** | **~20-30 min** | |

### Expected Performance:

Against RandomAgent: **85-95% win rate**
Against FixedStrategyAgent: **65-75% win rate**
Against CFRAgent: **50-60% win rate**

### Monitoring Progress:

The `--verbose` flag provides real-time updates:
- Episode number and progress
- Average reward (last N episodes)
- Win rate
- Epsilon value (exploration)
- Memory buffer size

**Example:**
```
[Stage2] Ep 50/100 | AvgR 18.50 | Win 68.00% | Eps 0.050 | Mem 400
```

---

## Production Training

**Purpose:** Championship-level agent training (4-8 hours total)

**When to use:** Final model for competition, deployment, serious play

### Command:

```bash
# Full production training
python scripts/train.py \
  --agent-type pokerbot \
  --mode production \
  --verbose \
  --report \
  --seed 42

# With optimization export (pruning and quantization)
python scripts/train.py \
  --agent-type pokerbot \
  --mode production \
  --verbose \
  --report \
  --optimize-export
```

### Configuration (Production Mode):

```json
{
  "stage1_cfr_iterations": 5000,
  "stage2_selfplay_episodes": 2000,
  "stage3_vicarious_episodes": 2000,
  "evaluation_interval": 100,
  "save_interval": 500,
  "batch_size": 32,
  "validation_hands": 500
}
```

### Timeline:

| Stage | Duration | Purpose |
|-------|----------|---------|
| Stage 1 (CFR) | ~30-45 min | Deep game theory |
| Stage 2 (Self-play) | ~2-4 hours | Advanced patterns |
| Stage 3 (Vicarious) | ~2-4 hours | Exploit mastery |
| Evaluation | ~15-30 min | Comprehensive testing |
| **Total** | **~4-8 hours** | |

### Expected Championship Performance:

Against RandomAgent: **95-99% win rate**
Against FixedStrategyAgent: **75-85% win rate**
Against CFRAgent: **55-65% win rate**
Against Previous Best: **>50% win rate** (required for promotion)

### Advanced Options:

**Custom Iterations:**
```bash
# Override specific stages
python scripts/train.py \
  --agent-type pokerbot \
  --mode production \
  --cfr-iterations 10000 \
  --episodes 5000 \
  --batch-size 64
```

**Resume Training:**
```bash
# Automatically resumes from best model if exists
python scripts/train.py --agent-type pokerbot --mode production
```

**Model Optimization:**
```bash
# Prune and quantize for deployment
python scripts/train.py \
  --agent-type pokerbot \
  --mode production \
  --optimize-export
```

This creates:
- `champion_current_pruned.keras` - Pruned model (50% sparsity)
- `champion_current.tflite` - TensorFlow Lite model (quantized)

---

## Model Evaluation

### Post-Training Validation

After training completes, validate your model:

```bash
# Validate trained model
python scripts/validate_training.py --model models/versions/champion_best --hands 500

# Compare models
python scripts/test_model_comparison.py \
  --model1 models/versions/champion_best \
  --model2 models/versions/champion_current \
  --hands 1000
```

### Testing Against Specific Opponents:

```bash
# Interactive testing
python scripts/play.py --agent pokerbot --opponent random
python scripts/play.py --agent pokerbot --opponent cfr
```

### Performance Metrics:

Check training metrics:
```bash
cat logs/training_metrics.json
```

Key metrics to review:
- `training_rewards` - Reward progression over episodes
- `win_rates` - Win rate progression
- `epsilon_values` - Exploration decay
- `episode_reward_mean` - Average reward per evaluation interval

### Visualize Strategy:

```bash
# Generate strategy visualization
python scripts/visualize_strategy.py \
  --model models/versions/champion_best \
  --output models/reports/strategy_viz.png
```

---

## Troubleshooting

### Common Issues

#### 1. Data Validation Fails

**Error:** `ValueError: DeepStacked training samples failed validation`

**Solution:**
```bash
# Check data files exist
ls -l data/deepstacked_training/samples/train_samples/
ls -l data/equity_tables/

# Regenerate if needed
python scripts/verify_pt_samples.py
python scripts/validate_data.py
```

#### 2. Out of Memory

**Error:** `RuntimeError: CUDA out of memory` or similar

**Solution:**
```bash
# Reduce batch size
python scripts/train.py --agent-type pokerbot --mode standard --batch-size 4

# Or use CPU only
python scripts/train.py --agent-type pokerbot --mode standard
```

#### 3. Import Errors

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify
python -c "import tensorflow, torch; print('OK')"
```

#### 4. Training Hangs

**Symptom:** Training stops making progress

**Solution:**
- Check system resources (CPU, memory)
- Reduce episode count or batch size
- Use Ctrl+C to interrupt, model saved at last checkpoint
- Resume with same command

#### 5. Poor Performance

**Symptom:** Low win rates after training

**Checklist:**
- ✓ Used production mode? (Standard is not championship-level)
- ✓ Trained for full duration? (Don't interrupt)
- ✓ Data validation passed?
- ✓ Sufficient iterations? (Check config)

**Solution:**
```bash
# Run production training
python scripts/train.py --agent-type pokerbot --mode production --verbose

# Or increase iterations
python scripts/train.py \
  --agent-type pokerbot \
  --mode production \
  --cfr-iterations 10000 \
  --episodes 5000
```

---

## Next Steps: Vision System

Once you have a trained championship-level PokerBot agent, you can proceed to integrate the vision/perception system for end-to-end automated play.

### Vision System Overview

**Components:**
1. **Screen Capture** - Captures poker client window
2. **Game State Detection** - Uses GPT-4 Vision API to parse game state
3. **Action Execution** - PyAutoGUI to execute actions
4. **Control Loop** - Coordinates everything

### Integration Steps:

#### 1. Verify Trained Agent

```bash
# Test agent in isolation
python examples/test_pokerbot.py
python examples/validate_pokerbot.py
```

Expected: All tests passing ✓

#### 2. Configure Vision System

Create `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
SCREEN_REGION_X=0
SCREEN_REGION_Y=0
SCREEN_REGION_WIDTH=1920
SCREEN_REGION_HEIGHT=1080
```

#### 3. Test Vision Components

```bash
# Test screen capture
python -c "
from src.vision import capture_screen
img = capture_screen()
print(f'Captured: {img.size}')
"

# Test game state detection
python scripts/test_vision.py --screenshot test_poker_table.png
```

#### 4. Run End-to-End System

```bash
# Interactive mode (manual confirmation)
python scripts/play.py --agent pokerbot --vision --interactive

# Automated mode (full automation)
python scripts/play.py --agent pokerbot --vision --auto
```

### Safety Measures:

**IMPORTANT:** When running automated play:

1. **Test Mode First:** Always test with play money or test accounts
2. **Monitor Carefully:** Watch the first 100+ hands
3. **Set Limits:** Configure stop-loss and time limits
4. **Emergency Stop:** Keep Ctrl+C ready to interrupt

### Vision System Architecture:

```
PokerBot End-to-End System:
┌─────────────────────────────────────┐
│ 1. Screen Capture (pyautogui)      │
│    ├─ Captures poker table         │
│    └─ Preprocesses image           │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ 2. GPT-4 Vision API                 │
│    ├─ Detects cards                │
│    ├─ Reads pot size               │
│    ├─ Identifies bets              │
│    └─ Parses game state            │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ 3. PokerBot Agent (Trained Model)   │
│    ├─ Processes game state         │
│    ├─ CFR/DQN/DeepStack decision   │
│    └─ Returns optimal action       │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ 4. Action Execution (pyautogui)     │
│    ├─ Clicks buttons               │
│    ├─ Enters bet amounts           │
│    └─ Confirms actions             │
└─────────────────────────────────────┘
```

### Development Workflow:

```bash
# Phase 1: Agent Training (COMPLETED with this guide)
python scripts/train.py --agent-type pokerbot --mode production

# Phase 2: Vision Testing (NEXT)
python scripts/test_vision.py
python scripts/calibrate_screen.py

# Phase 3: Integration Testing
python scripts/play.py --agent pokerbot --vision --interactive

# Phase 4: Live Testing (Careful!)
python scripts/play.py --agent pokerbot --vision --auto --test-mode

# Phase 5: Production Deployment
python scripts/play.py --agent pokerbot --vision --auto
```

---

## Additional Resources

### Testing Scripts:

```bash
# Test complete pipeline (no training)
python scripts/test_training_pipeline.py

# Profile performance
python scripts/profile_performance.py

# Monitor resources during training
python scripts/monitor_resources.py &
python scripts/train.py --agent-type pokerbot --mode production

# Hyperparameter tuning (Optuna)
python scripts/tune_hyperparams.py --trials 50
```

### Configuration Files:

- `scripts/config/smoketest.json` - Fast testing (1-2 min)
- `scripts/config/standard.json` - Development (20-30 min)
- `scripts/config/production.json` - Championship (4-8 hours)
- `scripts/config/training.json` - Custom template

### Documentation:

- `README.md` - Project overview
- `docs/MIGRATION_GUIDE.md` - Agent migration guide
- `docs/IMPLEMENTATION_SUMMARY.md` - Technical details
- `docs/IMPORT_UPDATE_SUMMARY.md` - Import changes

---

## Quick Reference

### Essential Commands:

```bash
# Smoketest (1-2 min)
python scripts/train.py --agent-type pokerbot --mode smoketest --verbose

# Standard training (20-30 min)
python scripts/train.py --agent-type pokerbot --mode standard --verbose --report

# Production training (4-8 hours)
python scripts/train.py --agent-type pokerbot --mode production --verbose --report --optimize-export

# Validate model
python scripts/validate_training.py --model models/versions/champion_best --hands 500

# Test agent
python examples/test_pokerbot.py
```

### Training Modes Comparison:

| Mode | CFR Iter | Episodes | Duration | Purpose |
|------|----------|----------|----------|---------|
| Smoketest | 10 | 10 + 10 | 1-2 min | Testing pipeline |
| Standard | 200 | 100 + 100 | 20-30 min | Development |
| Production | 5000 | 2000 + 2000 | 4-8 hours | Championship |

### File Locations:

- Models: `models/versions/champion_best.*`
- Checkpoints: `models/checkpoints/`
- Logs: `logs/training_metrics.json`
- Data: `data/deepstacked_training/`, `data/equity_tables/`
- Config: `scripts/config/*.json`

---

## Success Checklist

Before proceeding to vision system:

- [ ] All tests pass: `python examples/test_pokerbot.py`
- [ ] Smoketest completes successfully
- [ ] Standard training achieves >65% win rate vs Fixed
- [ ] Production training achieves >75% win rate vs Fixed
- [ ] Model saves correctly to `models/versions/champion_best.*`
- [ ] Validation script confirms performance
- [ ] Ready to integrate vision system

---

## Support and Contributing

For issues, questions, or contributions:
- Create an issue on GitHub
- Review existing documentation in `docs/`
- Check troubleshooting section above

---

**Training Your Championship PokerBot - Summary:**

1. ✓ Verify prerequisites (data, dependencies)
2. ✓ Run smoketest to verify pipeline
3. ✓ Run standard training for development
4. ✓ Run production training for championship model
5. ✓ Validate and evaluate your model
6. ✓ Proceed to vision system integration

**You are now ready to train a world-class PokerBot agent!**
