# Champion Agent Training Pipeline - User Guide

## Overview

This guide provides step-by-step instructions for training and validating the Champion Agent using our advanced progressive training pipeline with vicarious learning capabilities.

## What is the Training Pipeline?

The training pipeline implements a state-of-the-art, multi-stage approach to train the Champion Agent:

1. **Stage 1: CFR Warmup** - Builds game-theoretic foundation through Counterfactual Regret Minimization
2. **Stage 2: Self-Play** - Iterative improvement through playing against itself
3. **Stage 3: Vicarious Learning** - Learning from diverse opponents (Random, Fixed, CFR, DQN agents)

This progressive approach creates a well-rounded, adaptable agent that combines theoretical optimality with practical adaptability.

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space for models

### Python Dependencies
```bash
pip install numpy tensorflow
```

All other dependencies should already be installed as part of the project.

## Quick Start - Smoketest Mode

**Recommended for first-time users!** This mode runs a quick validation to ensure everything works correctly.

### Step 1: Run Smoketest Training

```bash
cd /path/to/pokerbot
python scripts/train_champion.py --mode smoketest
```

**What happens:**
- Stage 1: 100 CFR iterations (~30 seconds)
- Stage 2: 50 self-play episodes (~2-3 minutes)
- Stage 3: 50 vicarious learning episodes (~2-3 minutes)
- Total time: ~5-7 minutes

**Expected output:**
```
======================================================================
CHAMPION AGENT PROGRESSIVE TRAINING PIPELINE
======================================================================
Mode: SMOKETEST
Start time: 2024-01-15 10:30:00

STAGE 1: CFR WARMUP - Building game-theoretic foundation
----------------------------------------------------------------------
Training CFR component for 100 iterations...
✓ CFR warmup complete
  Total CFR iterations: 100
  Information sets learned: 45

STAGE 2: SELF-PLAY - Iterative self-improvement
----------------------------------------------------------------------
Self-play training for 50 episodes...
  Episode 25/50 - Avg Reward: 12.45 - Win Rate: 52.00% - Epsilon: 0.285 - Memory: 124
  Episode 50/50 - Avg Reward: 15.30 - Win Rate: 54.00% - Epsilon: 0.270 - Memory: 248
✓ Self-play complete - Final win rate: 54.00%

STAGE 3: VICARIOUS LEARNING - Learning from diverse opponents
----------------------------------------------------------------------
Vicarious learning for 50 episodes...
  Episode 25/50 - Epsilon: 0.256
    vs RANDOM: Win Rate 75.00%, Avg Reward 18.50
    vs FIXED: Win Rate 62.50%, Avg Reward 12.30
    vs CFR: Win Rate 48.00%, Avg Reward 5.20
  Episode 50/50 - Epsilon: 0.243
    vs RANDOM: Win Rate 74.00%, Avg Reward 19.10
    vs FIXED: Win Rate 64.00%, Avg Reward 13.50
    vs CFR: Win Rate 52.00%, Avg Reward 6.80
✓ Vicarious learning complete

FINAL EVALUATION
----------------------------------------------------------------------
Running final evaluation over 50 hands...
  Evaluating vs EvalRandom...
    Win Rate: 78.00%, Avg Reward: 21.45
  Evaluating vs EvalFixed...
    Win Rate: 66.00%, Avg Reward: 14.20
  Evaluating vs EvalCFR...
    Win Rate: 54.00%, Avg Reward: 7.30

======================================================================
TRAINING COMPLETE
======================================================================
Total time: 6.25 minutes
Total episodes: 100
Final model: models/champion_final
======================================================================
```

### Step 2: Validate the Training

After training completes, validate that the agent learned correctly:

```bash
python scripts/validate_training.py --model models/champion_final --hands 100
```

**Expected output:**
```
======================================================================
CHAMPION AGENT VALIDATION
======================================================================
Model: models/champion_final
Validation hands per opponent: 100

TEST 1: Agent State Validation
----------------------------------------------------------------------
  Epsilon value: 0.2430 ✓
  Memory size: 248 ✓
  CFR iterations: 100 ✓
  Information sets: 45 ✓
  DQN model: Present ✓

  Overall: PASSED ✓

TEST 2: Performance Against Baseline Agents
----------------------------------------------------------------------

  Testing against random...
    Win Rate: 76.00% ✓
    Avg Reward: 20.15
    Record: 76W - 24L

  Testing against fixed...
    Win Rate: 64.00% ✓
    Avg Reward: 13.80
    Record: 64W - 36L

  Testing against cfr...
    Win Rate: 52.00% ✓
    Avg Reward: 6.50
    Record: 52W - 48L

  Testing against untrained_champion...
    Win Rate: 58.00% ✓
    Avg Reward: 9.20
    Record: 58W - 42L

TEST 3: Decision Consistency Check
----------------------------------------------------------------------
  Testing decision consistency with same game state...
    Unique actions: 2
    Consistency score: 90.00%
    Status: ✓

TEST 4: Memory & Learning Validation
----------------------------------------------------------------------
  Checking learning indicators...
    Epsilon decay: 0.2430 ✓
    Memory usage: 12.40% ✓
    CFR convergence: 100 iterations ✓

    Overall: PASSED ✓

======================================================================
VALIDATION SUMMARY
======================================================================
Tests Passed: 13/13 (100.0%)
Status: EXCELLENT ✓✓✓
======================================================================
```

### Step 3: Test the Trained Agent

Now you can test your trained agent interactively:

```python
from src.agents import ChampionAgent
from src.game import Card, Rank, Suit

# Load trained agent
agent = ChampionAgent(name="TrainedChampion", use_pretrained=False)
agent.load_strategy("models/champion_final")

# Test a decision
hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
community_cards = []
pot = 100
current_bet = 20
player_stack = 1000

action, raise_amt = agent.choose_action(
    hole_cards, community_cards, pot,
    current_bet, player_stack, current_bet
)

print(f"Agent decision: {action}, Raise: {raise_amt}")
```

## Full Production Training

Once you've verified the smoketest works, you can run full production training:

### Step 1: Run Full Training

```bash
python scripts/train_champion.py --mode full
```

**What happens:**
- Stage 1: 5,000 CFR iterations (~10-15 minutes)
- Stage 2: 2,000 self-play episodes (~45-60 minutes)
- Stage 3: 3,000 vicarious learning episodes (~60-90 minutes)
- Total time: ~2-3 hours

**Training will automatically:**
- Save checkpoints every 500 episodes
- Print progress every 100 episodes
- Evaluate performance periodically
- Generate comprehensive training reports

### Step 2: Monitor Training Progress

The training script saves checkpoints at regular intervals. You can monitor progress by checking:

1. **Console output** - Real-time statistics
2. **Model checkpoints** - Saved in `models/` directory
3. **Training logs** - Saved in `logs/` directory

### Step 3: Validate Full Training

After completion, validate with more hands:

```bash
python scripts/validate_training.py --model models/champion_final --hands 500
```

## Advanced Usage

### Custom Training Parameters

```bash
# Custom number of episodes
python scripts/train_champion.py --mode smoketest --episodes 200

# Custom CFR iterations
python scripts/train_champion.py --mode smoketest --cfr-iterations 500

# Custom model directory
python scripts/train_champion.py --mode smoketest --model-dir my_models/

# Combine options
python scripts/train_champion.py --mode full --episodes 5000 --cfr-iterations 10000
```

### Resume from Checkpoint

If training is interrupted, resume from a checkpoint:

```bash
python scripts/train_champion.py --mode full --resume models/champion_checkpoint_stage2_ep1000
```

### Analyzing Training Results

After training, several files are created:

1. **Model files:**
   - `models/champion_final.cfr` - CFR strategy
   - `models/champion_final.dqn` - DQN neural network
   - `models/champion_final_metadata.json` - Training metadata

2. **Training report:**
   - `logs/training_report_YYYYMMDD_HHMMSS.txt` - Comprehensive report

3. **Validation results:**
   - `models/champion_final_validation.json` - Validation metrics

View the training report:
```bash
cat logs/training_report_*.txt
```

View validation results:
```bash
cat models/champion_final_validation.json
```

## Understanding the Metrics

### Training Metrics

- **Epsilon**: Exploration rate (decreases over time as agent learns)
- **Win Rate**: Percentage of hands won
- **Avg Reward**: Average chips won/lost per hand
- **Memory**: Number of experiences stored for learning

### Validation Metrics

- **Win Rate vs Baseline**: Should be >50% against most agents
- **Consistency Score**: Should be >50% (higher = more consistent decisions)
- **Memory Usage**: Should be >10% after training
- **CFR Convergence**: Number of CFR iterations completed

### Success Criteria

A successfully trained agent should have:
- ✓ Win rate >70% vs Random agents
- ✓ Win rate >60% vs Fixed strategy agents  
- ✓ Win rate >45% vs CFR agents
- ✓ Epsilon <0.3 (has learned to exploit knowledge)
- ✓ Memory >100 experiences
- ✓ CFR iterations >50

## Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow
```

### Issue: Training is very slow

**Solutions:**
- Use smoketest mode for faster validation
- Reduce number of episodes: `--episodes 100`
- Ensure TensorFlow is using CPU efficiently (GPU not required)

### Issue: Agent performs poorly after training

**Possible causes:**
1. Training interrupted too early
2. Insufficient episodes
3. Check validation results for specific weaknesses

**Solutions:**
- Run full training mode (not smoketest)
- Increase episodes: `--episodes 5000`
- Validate and check which opponents are problematic

### Issue: Memory errors during training

**Solutions:**
- Reduce batch size in code (default: 32)
- Close other applications
- Use smaller CFR iterations

### Issue: Checkpoint files not found

**Check:**
```bash
ls -la models/
```

Should see files like:
- `champion_final.cfr`
- `champion_final.dqn`
- `champion_final_metadata.json`

If missing, training may have failed. Check console output for errors.

## Best Practices

### 1. Always Start with Smoketest

Run smoketest mode first to verify everything works:
```bash
python scripts/train_champion.py --mode smoketest
```

### 2. Validate After Training

Always validate to ensure training was effective:
```bash
python scripts/validate_training.py --model models/champion_final
```

### 3. Save Multiple Checkpoints

Keep checkpoints from different stages:
```bash
# Stage 2 checkpoint
models/champion_checkpoint_stage2_ep1000.cfr

# Stage 3 checkpoint  
models/champion_checkpoint_stage3_ep2000.cfr

# Final model
models/champion_final.cfr
```

### 4. Compare Performance

Compare trained vs untrained agents:
```python
from src.agents import ChampionAgent
from src.evaluation import Evaluator

# Load agents
trained = ChampionAgent(name="Trained", use_pretrained=False)
trained.load_strategy("models/champion_final")

untrained = ChampionAgent(name="Untrained", use_pretrained=False)

# Evaluate
evaluator = Evaluator([trained, untrained])
results = evaluator.evaluate_agents(num_hands=100)
```

### 5. Monitor System Resources

During full training, monitor:
- CPU usage (should be 50-100%)
- Memory usage (should be <4GB)
- Disk space (need ~1GB free)

## Next Steps

After successful training:

1. **Integrate into your application** - Use the trained agent in your poker bot
2. **Continue training** - Resume and train for more episodes
3. **Experiment with opponents** - Create custom opponents for specialized training
4. **Tune hyperparameters** - Adjust learning rates, epsilon decay, etc.
5. **Benchmark performance** - Test against other poker AIs

## Frequently Asked Questions

### Q: How long does training take?

**A:** 
- Smoketest: 5-7 minutes
- Full training: 2-3 hours

### Q: Can I train on CPU only?

**A:** Yes! The pipeline works fine on CPU. GPU is not required.

### Q: How much improvement should I expect?

**A:** A successfully trained agent should:
- Beat random agents 75%+ of the time
- Beat fixed strategy agents 60%+ of the time
- Beat CFR agents 50%+ of the time (competitive)

### Q: Can I interrupt and resume training?

**A:** Yes! Use Ctrl+C to stop, then resume:
```bash
python scripts/train_champion.py --resume models/champion_checkpoint_stage2_ep500
```

### Q: What if validation fails?

**A:** Check:
1. Training completed successfully (no errors)
2. Model files exist in `models/` directory
3. Run full training instead of smoketest
4. Increase training episodes

### Q: How do I know if my agent is learning?

**A:** Look for:
- Decreasing epsilon values
- Increasing win rates over time
- Growing memory size
- Positive performance in validation

## Support

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review console output for error messages
3. Verify all dependencies are installed
4. Try smoketest mode first
5. Check validation results for specific issues

## Summary

**Quick Start Checklist:**
- [ ] Install dependencies (`pip install numpy tensorflow`)
- [ ] Run smoketest (`python scripts/train_champion.py --mode smoketest`)
- [ ] Validate training (`python scripts/validate_training.py --model models/champion_final`)
- [ ] Review results (check win rates and metrics)
- [ ] Run full training if smoketest succeeds
- [ ] Test your trained agent in action!

**You're now ready to train a championship-level poker AI! 🏆**
