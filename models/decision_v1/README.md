# Decision Model (Strategist)

This directory contains the trained value network weights for decision making.

## Expected Contents
- `value_network.pt` - PyTorch model weights
- `config.json` - Model configuration (optional)
- `training_log.json` - Training metrics (optional)

## Model Architecture
Based on DeepStack paper (Section S3):
- Input: [player_range, opponent_range, pot_size]
- Hidden: 7 layers × 500 units with PReLU activation
- Output: [player_values, opponent_values]

## Training
The value network is trained on CFR-generated data:
1. Generate training samples using self-play CFR
2. Train network to predict counterfactual values
3. Validate against held-out CFR solutions

See `docs/TRAINING_GUIDE.md` for detailed instructions.
