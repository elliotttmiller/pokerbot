# Training Pipeline Model Management

This document describes the improved model management system in the Champion Agent training pipeline.

## Overview

The training pipeline now implements an intelligent model management system that:
- **Prevents file proliferation** by using organized directory structure
- **Ensures continuous improvement** by comparing new models against previous best
- **Automatically promotes** better models while preserving the best performer
- **Always starts from champion baseline** using pre-trained DeepStack models

## Directory Structure

The training pipeline uses an organized directory structure:

```
models/
├── versions/          # Production-ready model versions
│   ├── champion_current.*       # Latest trained model
│   ├── champion_best.*          # Best performing model
│   └── champion_best_comparison.json
└── checkpoints/       # Training checkpoints
    └── champion_checkpoint.*    # Single checkpoint per run (overwrites)
```

## Model Files

### Primary Models (models/versions/)

#### `champion_current`
- Latest trained model from the most recent training session
- **Overwritten** with each training run
- Used for experimentation and iteration
- Location: `models/versions/champion_current.*`
- Files: `.cfr`, `.keras`, `_metadata.json`

#### `champion_best`
- Best performing model across all training sessions
- **Only updated** when a new model outperforms it in head-to-head comparison
- This is the model you should use for production/competition
- Location: `models/versions/champion_best.*`
- Files: `.cfr`, `.keras`, `_metadata.json`

#### `champion_best_comparison.json`
- Contains results of the most recent comparison between champion_current and champion_best
- Includes win rate, average reward, and promotion decision
- Location: `models/versions/champion_best_comparison.json`

### Training Checkpoints (models/checkpoints/)

- `champion_checkpoint.*` - Single checkpoint representing the overall training run
- Overwrites with each training session
- Serves as recovery point if training is interrupted
- Location: `models/checkpoints/champion_checkpoint.*`

## Pre-trained Champion Models

The agent **always** starts with world-class pre-trained models:
- **DeepStack value networks** for champion-level value estimation
- **Preflop equity tables** for optimal early-game decisions
- **Never starts from scratch** - iterates from champion baseline
- Uses `use_pretrained=True` and `use_deepstack=True` by default

This ensures every training session builds upon proven champion-level knowledge.

## Training Workflow

### 1. Training Execution
```bash
python scripts/train_champion.py --mode smoketest
```

The training pipeline goes through three stages:
1. **Stage 1: CFR Warmup** - Builds game-theoretic foundation
2. **Stage 2: Self-Play** - Iterative improvement against itself
3. **Stage 3: Vicarious Learning with Adaptive Curriculum** - Learning from diverse opponents, prioritizing challenging ones

### 2. Model Saving
After training completes, the new model is saved as `models/versions/champion_current`, overwriting the previous version.

### 3. Comparison Against Previous Best
The pipeline automatically:
- Loads the previous `champion_best` model from `models/versions/` (if it exists)
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
Initializing Champion Agent with pre-trained models...
Note: This agent starts with world-class champion knowledge and iterates from there.

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

### 1. Organized Directory Structure
- Clean separation: versions in `models/versions/`, checkpoints in `models/checkpoints/`
- No clutter in models root directory
- Easy to identify production models vs training checkpoints

### 2. Champion Baseline
- Always starts with world-class pre-trained models (DeepStack + equity tables)
- Never trains from scratch - iterates from proven champion knowledge
- Guaranteed strong baseline performance

### 3. Guaranteed Improvement
- `champion_best` always contains the best performing model
- New models must prove themselves in head-to-head competition
- No risk of regression by accidentally using a worse model

### 4. Enhanced Learning
- Adaptive curriculum in vicarious learning prioritizes challenging opponents
- Double training iterations for better learning from experiences
- Focuses training where agent needs improvement most

### 5. Experimentation Friendly
- `champion_current` allows you to experiment with different training parameters
- If experiment doesn't work out, `champion_best` remains unchanged
- Can analyze failed experiments by comparing `champion_current` vs `champion_best`

### 6. Clear Progression Tracking
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
python scripts/train_champion.py --resume models/versions/champion_best
```

### Resume from Current Model
```bash
# Continue from last training (even if not promoted)
python scripts/train_champion.py --resume models/versions/champion_current
```

### Compare Results
```bash
# View comparison results
cat models/versions/champion_best_comparison.json

# View training report
cat logs/training_report_*.txt | tail -50
```

## Testing

A test script is provided to validate the model management system:

```bash
python scripts/test_model_comparison.py
```

This verifies:
- Correct directory structure (versions/ and checkpoints/)
- Comparison results are saved
- Metadata consistency
- No unexpected timestamped files

## Migration from Old System

If you have old model files:

1. **Identify your best model** - Look at validation results or timestamps
2. **Copy to new directory structure**:
   ```bash
   # Create directories if they don't exist
   mkdir -p models/versions models/checkpoints
   
   # Copy best model to versions directory
   cp models/old_model_name.cfr models/versions/champion_best.cfr
   cp models/old_model_name.keras models/versions/champion_best.keras
   cp models/old_model_name_metadata.json models/versions/champion_best_metadata.json
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

## Advanced Features

### Adaptive Curriculum Learning
The vicarious learning stage now uses adaptive curriculum:
- **Early episodes**: Round-robin through all opponent types to gather initial data
- **Later episodes**: Prioritizes challenging opponents based on difficulty scores
- **Difficulty calculation**: `1.0 - win_rate` (lower win rate = higher difficulty = more practice)
- **Selection probability**: Uses softmax over difficulty scores

This ensures the agent focuses training on its weaknesses, leading to faster improvement.

### Enhanced Training
- **Double replay iterations**: During vicarious learning, experiences are replayed twice per episode
- **Better learning**: More training iterations help the agent better internalize patterns from challenging opponents

