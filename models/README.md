# Models Directory

This directory stores trained Champion Agent models.

## File Types

- `*.cfr` - CFR (Counterfactual Regret Minimization) strategy files
- `*.keras` - DQN neural network models  
- `*_metadata.json` - Training metadata and statistics
- `*_validation.json` - Validation test results

## Generated Files

When you run training, these files are automatically created:

### During Training
- `champion_checkpoint_stage1.*` - After Stage 1 (CFR warmup)
- `champion_checkpoint_stage2_epN.*` - Stage 2 checkpoints (every N episodes)
- `champion_checkpoint_stage3_epN.*` - Stage 3 checkpoints (every N episodes)

### After Training
- `champion_final.cfr` - Final CFR strategy
- `champion_final.keras` - Final DQN model
- `champion_final_metadata.json` - Final training statistics
- `champion_final_validation.json` - Validation results (after running validation script)

## Usage

Load a trained model:

```python
from src.agents import ChampionAgent

# Load trained agent
agent = ChampionAgent(name="TrainedChampion", use_pretrained=False)
agent.load_strategy("models/champion_final")

# Use the agent
action, raise_amt = agent.choose_action(hole_cards, community_cards, pot, bet, stack, opp_bet)
```

## Note

Model files are not committed to git (see `.gitignore`). You need to train your own models using:

```bash
python scripts/train_champion.py --mode smoketest
```

See `docs/TRAINING_GUIDE.md` for complete instructions.
