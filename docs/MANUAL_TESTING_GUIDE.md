# Champion Agent Training Pipeline - Manual Testing & Validation Guide

## Step-by-Step Testing Guide

This guide walks you through manually testing and validating the training pipeline to ensure it's working correctly.

---

## Prerequisites Check

Before starting, verify:
```bash
# 1. Check Python version (3.8+)
python --version

# 2. Check dependencies
python -c "import numpy, tensorflow; print('Dependencies OK')"

# 3. Navigate to project root
cd /path/to/pokerbot
```

---

## Step 1: Run Automated Test Suite (5 minutes)

First, run the comprehensive automated test to verify everything works:

```bash
python scripts/test_training_pipeline.py
```

**Expected Output:**
```
======================================================================
CHAMPION AGENT TRAINING PIPELINE - END-TO-END TEST
======================================================================

TEST 1: Testing imports...
  ✓ All imports successful

TEST 2: Testing agent creation...
  ✓ Agent created successfully
    State size: 60
    Action size: 3
    Epsilon: 0.3

TEST 3: Testing training pipeline...
  Running minimal training (this may take 1-2 minutes)...
  ✓ Training completed successfully
    ✓ models/champion_final.cfr created
    ✓ models/champion_final.keras created

TEST 4: Testing model loading...
  ✓ Model loaded successfully
    CFR iterations: 10
    Information sets: 38

TEST 5: Testing agent decision making...
  ✓ Agent made decision successfully
    Action: <Action.RAISE>
    Raise amount: 60

TEST 6: Testing validation script...
  Running validation (this may take 30-60 seconds)...
  ✓ Validation completed successfully
    ✓ Validation results saved

======================================================================
TEST SUMMARY
======================================================================
  Imports: ✓ PASSED
  Agent Creation: ✓ PASSED
  Training Pipeline: ✓ PASSED
  Model Loading: ✓ PASSED
  Decision Making: ✓ PASSED
  Validation: ✓ PASSED

Total: 6/6 tests passed
Time: 42.1 seconds
======================================================================

🎉 ALL TESTS PASSED! Training pipeline is working correctly.
```

**✅ If all tests pass, proceed to Step 2.**

**❌ If any tests fail:**
- Check error messages
- Verify dependencies are installed: `pip install numpy tensorflow`
- Review `docs/TRAINING_GUIDE.md` Troubleshooting section

---

## Step 2: Run Manual Smoketest Training (5-7 minutes)

Now run a manual smoketest to see the training process in detail:

```bash
# Clean previous runs
rm -rf models/champion* logs/*

# Run smoketest
python scripts/train_champion.py --mode smoketest
```

**What to Watch For:**

### Stage 1: CFR Warmup
```
STAGE 1: CFR WARMUP - Building game-theoretic foundation
----------------------------------------------------------------------
Training CFR component for 100 iterations...
✓ CFR warmup complete
  Total CFR iterations: 100
  Information sets learned: 45-60
```
**✅ Expected:** ~30 seconds, 40-60 information sets learned

### Stage 2: Self-Play
```
STAGE 2: SELF-PLAY - Iterative self-improvement
----------------------------------------------------------------------
Self-play training for 50 episodes...
  Episode 25/50 - Avg Reward: 10.00-20.00 - Win Rate: 45-55% - Epsilon: 0.28-0.30
  Episode 50/50 - Avg Reward: 10.00-20.00 - Win Rate: 45-55% - Epsilon: 0.26-0.28
✓ Self-play complete - Final win rate: 45-55%
```
**✅ Expected:** ~2-3 minutes, 45-55% win rate

### Stage 3: Vicarious Learning
```
STAGE 3: VICARIOUS LEARNING - Learning from diverse opponents
----------------------------------------------------------------------
Vicarious learning for 50 episodes...
  Episode 25/50 - Epsilon: 0.25
    vs RANDOM: Win Rate 70-80%, Avg Reward 15-25
    vs FIXED: Win Rate 55-65%, Avg Reward 8-15
    vs CFR: Win Rate 45-55%, Avg Reward 3-10
✓ Vicarious learning complete
```
**✅ Expected:** ~2-3 minutes, improving performance vs different opponents

### Final Evaluation
```
FINAL EVALUATION
----------------------------------------------------------------------
Running final evaluation over 50 hands...
  Evaluating vs EvalRandom...
    Win Rate: 70-80%, Avg Reward: 15-25
  Evaluating vs EvalFixed...
    Win Rate: 55-65%, Avg Reward: 10-18
  Evaluating vs EvalCFR...
    Win Rate: 45-60%, Avg Reward: 5-12
```
**✅ Expected:** Agent should beat Random 70%+, Fixed 55%+, CFR 45%+

### Training Complete
```
======================================================================
TRAINING COMPLETE
======================================================================
Total time: 5-7 minutes
Total episodes: 100
Final model: models/champion_final
======================================================================
```

---

## Step 3: Validate the Trained Model (2-3 minutes)

Now validate that the agent learned correctly:

```bash
python scripts/validate_training.py --model models/champion_final --hands 100
```

**Expected Output:**

### Test 1: Agent State
```
TEST 1: Agent State Validation
----------------------------------------------------------------------
  Epsilon value: 0.24-0.28 ✓
  Memory size: 100-300 ✓
  CFR iterations: 100 ✓
  Information sets: 40-60 ✓
  DQN model: Present ✓

  Overall: PASSED ✓
```

### Test 2: Performance vs Baselines
```
TEST 2: Performance Against Baseline Agents
----------------------------------------------------------------------

  Testing against random...
    Win Rate: 70-80% ✓
    Avg Reward: 15-25
    Record: 70-80W - 20-30L

  Testing against fixed...
    Win Rate: 55-65% ✓
    Avg Reward: 8-15
    Record: 55-65W - 35-45L

  Testing against cfr...
    Win Rate: 45-55% ✓
    Avg Reward: 3-10
    Record: 45-55W - 45-55L
```

### Test 3: Consistency
```
TEST 3: Decision Consistency Check
----------------------------------------------------------------------
  Testing decision consistency with same game state...
    Unique actions: 1-3
    Consistency score: 80-100%
    Status: ✓
```

### Test 4: Learning Indicators
```
TEST 4: Memory & Learning Validation
----------------------------------------------------------------------
  Checking learning indicators...
    Epsilon decay: 0.24-0.28 ✓
    Memory usage: 10-30% ✓
    CFR convergence: 100 iterations ✓

    Overall: PASSED ✓
```

### Validation Summary
```
======================================================================
VALIDATION SUMMARY
======================================================================
Tests Passed: 11-13/13 (85-100%)
Status: EXCELLENT ✓✓✓ or GOOD ✓✓
======================================================================
```

**✅ Expected:** 85%+ pass rate, GOOD or EXCELLENT status

---

## Step 4: Inspect Generated Files

Check that all expected files were created:

```bash
# View model files
ls -lh models/
```

**Expected Files:**
```
champion_checkpoint_stage1.cfr         (~8KB)
champion_checkpoint_stage1.keras       (~300KB)
champion_checkpoint_stage1_metadata.json (~500B)
champion_final.cfr                     (~8KB)
champion_final.keras                   (~900KB)
champion_final_metadata.json           (~1KB)
champion_final_validation.json         (~800B)
```

```bash
# View log files
ls -lh logs/
```

**Expected Files:**
```
training_report_*.txt                  (~800B)
pokerbot_*.log                         (~3KB each)
```

```bash
# Read training report
cat logs/training_report_*.txt
```

**Expected Content:**
```
======================================================================
CHAMPION AGENT TRAINING REPORT
======================================================================

Training Mode: SMOKETEST
Total Training Time: 5-7 minutes
Total Episodes: 100
Total Hands Played: 100

Stage Summary:
  Stage 1 (CFR Warmup): 100 iterations
  Stage 2 (Self-Play): 50 episodes
  Stage 3 (Vicarious): 50 episodes

Final Agent State:
  Epsilon: 0.24-0.28
  Memory Size: 100-300
  CFR Iterations: 100
  Information Sets: 40-60

Final Evaluation Results:
  vs EvalRandom: Win Rate: 70-80%, Avg Reward: 15-25
  vs EvalFixed: Win Rate: 55-65%, Avg Reward: 8-15
  vs EvalCFR: Win Rate: 45-60%, Avg Reward: 5-12
```

---

## Step 5: Test Trained Agent Interactively

Load and test your trained agent:

```bash
python
```

```python
from src.agents import ChampionAgent
from src.game import Card, Rank, Suit

# Load trained agent
agent = ChampionAgent(name="TrainedChampion", use_pretrained=False)
agent.load_strategy("models/champion_final")

# Test decision with strong hand (Ace-King)
hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
action, raise_amt = agent.choose_action(
    hole_cards=hole_cards,
    community_cards=[],
    pot=100,
    current_bet=20,
    player_stack=1000,
    opponent_bet=20
)
print(f"Strong hand decision: {action}, Raise: {raise_amt}")
# Expected: Likely RAISE with moderate amount (40-100)

# Test decision with weak hand (7-2)
hole_cards = [Card(Rank.SEVEN, Suit.SPADES), Card(Rank.TWO, Suit.HEARTS)]
action, raise_amt = agent.choose_action(
    hole_cards=hole_cards,
    community_cards=[],
    pot=100,
    current_bet=50,
    player_stack=1000,
    opponent_bet=50
)
print(f"Weak hand decision: {action}, Raise: {raise_amt}")
# Expected: Likely FOLD or CHECK

# Check agent state
print(f"\nAgent State:")
print(f"  CFR Iterations: {agent.cfr.iterations}")
print(f"  Memory Size: {len(agent.memory)}")
print(f"  Epsilon: {agent.epsilon}")
print(f"  Information Sets: {len(agent.cfr.infosets)}")
```

**✅ Expected Behavior:**
- Strong hands → Aggressive play (RAISE)
- Weak hands → Conservative play (FOLD/CHECK)
- Agent has learned (CFR iterations > 0, memory > 0)

---

## Step 6: Run Full Production Training (Optional, 2-3 hours)

If smoketest works perfectly, run full production training:

```bash
# Clean previous runs
rm -rf models/champion* logs/*.txt

# Run full training (2-3 hours)
python scripts/train_champion.py --mode full
```

**Expected Results (Full Training):**
- Stage 1: 5,000 CFR iterations (~15 minutes)
- Stage 2: 2,000 self-play episodes (~60 minutes)
- Stage 3: 3,000 vicarious episodes (~90 minutes)
- **Total time: 2-3 hours**

**Expected Performance:**
- vs Random: 75-85% win rate
- vs Fixed: 65-75% win rate
- vs CFR: 50-60% win rate

Then validate:
```bash
python scripts/validate_training.py --model models/champion_final --hands 500
```

---

## Troubleshooting Common Issues

### Issue: Import errors
**Solution:**
```bash
pip install numpy tensorflow
```

### Issue: "No module named 'src'"
**Solution:**
```bash
export PYTHONPATH=/path/to/pokerbot:$PYTHONPATH
```

### Issue: Training very slow
**Solution:**
- Use smoketest mode for testing
- Reduce episodes: `--episodes 50`
- Check CPU usage (should be 50-100%)

### Issue: Low win rates
**Solution:**
- Run full training mode (not smoketest)
- Smoketest is just for validation, not strong performance
- Full training needed for competitive agent

### Issue: Validation fails
**Check:**
1. Training completed without errors
2. Model files exist in `models/`
3. Run with more hands: `--hands 200`

---

## Success Criteria Checklist

After completing all steps, verify:

- [ ] Automated tests pass (6/6 tests)
- [ ] Smoketest training completes in 5-7 minutes
- [ ] All 3 training stages execute successfully
- [ ] Model files created (`.cfr`, `.keras`, `.json`)
- [ ] Training report generated in `logs/`
- [ ] Validation shows 80%+ pass rate
- [ ] Agent beats Random agents 70%+
- [ ] Agent beats Fixed agents 55%+
- [ ] Agent beats CFR agents 45%+
- [ ] Epsilon decayed from 0.3 to ~0.25
- [ ] Memory accumulated (100+ experiences)
- [ ] CFR learned 40+ information sets
- [ ] Interactive testing shows sensible decisions

**✅ If all items checked, your training pipeline is fully functional!**

---

## Next Steps

1. **Integrate into your application** - Use the trained agent
2. **Continue training** - Run full mode for better performance
3. **Experiment** - Try different hyperparameters
4. **Benchmark** - Compare against other poker AIs
5. **Deploy** - Use in your poker bot application

---

## Support Resources

- **Quick Start:** `docs/TRAINING_QUICKSTART.md`
- **Full Guide:** `docs/TRAINING_GUIDE.md`
- **Code:** `scripts/train_champion.py`, `scripts/validate_training.py`
- **Tests:** `scripts/test_training_pipeline.py`

---

**Congratulations! You now have a fully functional, state-of-the-art poker AI training pipeline! 🎉**
