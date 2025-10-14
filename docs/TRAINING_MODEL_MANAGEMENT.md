# Training Pipeline Model Management

This document describes the improved model management system in the Champion Agent training pipeline.

## Overview

The training pipeline now implements an intelligent model management system that:
- **Prevents file proliferation** by using consistent filenames
- **Ensures continuous improvement** by comparing new models against previous best
- **Automatically promotes** better models while preserving the best performer

## Model Files

### Primary Models

#### `champion_current`
- Latest trained model from the most recent training session
- **Overwritten** with each training run
- Used for experimentation and iteration
- Files: `.cfr`, `.keras`, `_metadata.json`

#### `champion_best`
- Best performing model across all training sessions
- **Only updated** when a new model outperforms it in head-to-head comparison
- This is the model you should use for production/competition
- Files: `.cfr`, `.keras`, `_metadata.json`

#### `champion_best_comparison.json`
- Contains results of the most recent comparison between champion_current and champion_best
- Includes win rate, average reward, and promotion decision

### Stage Checkpoints

- `champion_checkpoint_stage1` - After CFR warmup
- `champion_checkpoint_stage2` - After self-play (if saved)
- `champion_checkpoint_stage3` - After vicarious learning (if saved)

These are overwritten each training session and serve as recovery points during training.

## Training Workflow

### 1. Training Execution
```bash
python scripts/train_champion.py --mode smoketest
```

The training pipeline goes through three stages:
1. **Stage 1: CFR Warmup** - Builds game-theoretic foundation
2. **Stage 2: Self-Play** - Iterative improvement against itself
3. **Stage 3: Vicarious Learning** - Learning from diverse opponents

### 2. Model Saving
After training completes, the new model is saved as `champion_current`, overwriting the previous version.

### 3. Comparison Against Previous Best
The pipeline automatically:
- Loads the previous `champion_best` model (if it exists)
- Plays 50-200 hands between `champion_current` and `champion_best` (50 for smoketest, 200 for full mode)
- Calculates win rate and average reward

### 4. Promotion Decision
- **If win rate > 50%**: New model is promoted to `champion_best`
- **If win rate ≤ 50%**: Previous `champion_best` is preserved, new model stays as `champion_current`

### 5. Reporting
A detailed training report is generated including:
- Training statistics
- Final evaluation results
- Comparison results vs previous best
- Promotion decision

## Example Output

### First Training (No Previous Best)
```
MODEL COMPARISON & VALIDATION
----------------------------------------------------------------------
No previous best model found - current model will become the best
✓ New model outperforms previous best - promoting to champion_best
  Model promotion complete!
```

### Subsequent Training (Comparison)
```
MODEL COMPARISON & VALIDATION
----------------------------------------------------------------------
Comparing current model vs previous best over 200 hands...
  Comparison results:
    Win Rate: 62.00%
    Avg Reward: 15.30
    Result: ✓ NEW MODEL IS BETTER
✓ New model outperforms previous best - promoting to champion_best
  Model promotion complete!
```

### When New Model Doesn't Improve
```
MODEL COMPARISON & VALIDATION
----------------------------------------------------------------------
Comparing current model vs previous best over 200 hands...
  Comparison results:
    Win Rate: 38.00%
    Avg Reward: -8.45
    Result: ✗ Previous model is still better
✗ New model does not outperform previous best - keeping previous champion_best
  Current model saved as champion_current for analysis
```

## Benefits

### 1. No File Proliferation
- No more `champion_checkpoint_ep100`, `champion_checkpoint_ep200`, etc.
- Models directory stays clean and organized
- Easy to identify which model to use

### 2. Guaranteed Improvement
- `champion_best` always contains the best performing model
- New models must prove themselves in head-to-head competition
- No risk of regression by accidentally using a worse model

### 3. Experimentation Friendly
- `champion_current` allows you to experiment with different training parameters
- If experiment doesn't work out, `champion_best` remains unchanged
- Can analyze failed experiments by comparing `champion_current` vs `champion_best`

### 4. Clear Progression Tracking
- Each training report documents whether improvement occurred
- `champion_best_comparison.json` provides quantitative metrics
- Easy to track model evolution over time

## Usage Patterns

### Standard Training
```bash
# Train and automatically compare
python scripts/train_champion.py --mode full
```

### Resume from Best Model
```bash
# Continue training from your best model
python scripts/train_champion.py --resume models/champion_best
```

### Resume from Current Model
```bash
# Continue from last training (even if not promoted)
python scripts/train_champion.py --resume models/champion_current
```

### Compare Results
```bash
# View comparison results
cat models/champion_best_comparison.json

# View training report
cat logs/training_report_*.txt | tail -50
```

## Testing

A test script is provided to validate the model management system:

```bash
python scripts/test_model_comparison.py
```

This verifies:
- Correct file structure
- Comparison results are saved
- Metadata consistency
- No unexpected timestamped files

## Migration from Old System

If you have old model files with timestamps or episode numbers:

1. **Identify your best model** - Look at validation results or timestamps
2. **Copy to champion_best**:
   ```bash
   cp models/old_model_name.cfr models/champion_best.cfr
   cp models/old_model_name.keras models/champion_best.keras
   cp models/old_model_name_metadata.json models/champion_best_metadata.json
   ```
3. **Run new training** - The system will automatically compare and promote as appropriate

## Configuration

The number of comparison hands can be adjusted in `TrainingConfig`:
- Smoketest mode: 50 hands
- Full mode: 200 hands

This can be modified in `scripts/train_champion.py`:
```python
class TrainingConfig:
    def __init__(self, mode: str = "smoketest"):
        # ...
        self.validation_hands = 50  # for smoketest
        # or
        self.validation_hands = 200  # for full
```
