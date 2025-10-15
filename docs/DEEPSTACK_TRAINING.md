# DeepStack Neural Network Training Guide

## Overview

This guide explains how to train the DeepStack value network using the optimized training pipeline. The value network is a critical component of DeepStack's continual re-solving algorithm, used to estimate counterfactual values at lookahead leaves.

## Quick Start

### Basic Training

```bash
# Train with default configuration (50 epochs, CPU)
python scripts/train_deepstack.py

# Train with GPU acceleration
python scripts/train_deepstack.py --use-gpu

# Quick smoke test (5 epochs)
python scripts/train_deepstack.py --epochs 5 --no-gpu
```

### Custom Training

```bash
# Custom hyperparameters
python scripts/train_deepstack.py --epochs 100 --batch-size 64 --lr 0.0005

# Use a different config file
python scripts/train_deepstack.py --config scripts/config/my_training.json

# Override data path
python scripts/train_deepstack.py --data-path data/my_training_samples
```

## Configuration

### Training Configuration File

The default configuration is in `scripts/config/training.json`:

```json
{
  "num_buckets": 36,
  "bucket_count": 36,
  "bet_sizing": [1, 2],
  "hidden_sizes": [256, 256, 128],
  "activation": "prelu",
  "data_path": "data/deepstacked_training/samples/train_samples",
  "batch_size": 32,
  "use_gpu": true,
  "lr": 0.001,
  "epochs": 50,
  "model_save_dir": "models/pretrained"
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `num_buckets` | int | Number of hand buckets/abstractions | 36 |
| `hidden_sizes` | list[int] | Hidden layer sizes | [256, 256, 128] |
| `activation` | str | Activation function ('relu' or 'prelu') | 'prelu' |
| `data_path` | str | Path to training samples directory | data/deepstacked_training/samples/train_samples |
| `batch_size` | int | Training batch size | 32 |
| `use_gpu` | bool | Use GPU acceleration | true |
| `lr` | float | Learning rate | 0.001 |
| `epochs` | int | Maximum training epochs | 50 |
| `model_save_dir` | str | Directory to save models | models/pretrained |

## Training Data Format

The training data should be in PyTorch tensor format (.pt files):

```
data/deepstacked_training/samples/train_samples/
├── train_inputs.pt    # Training inputs [N, 2*buckets + 1]
├── train_targets.pt   # Training targets [N, 2*buckets]
├── train_mask.pt      # Training masks [N, buckets]
├── valid_inputs.pt    # Validation inputs
├── valid_targets.pt   # Validation targets
└── valid_mask.pt      # Validation masks
```

### Data Dimensions

- **Inputs**: `[batch_size, 2 * num_buckets + 1]`
  - First `num_buckets` values: Player's range
  - Next `num_buckets` values: Opponent's range
  - Last value: Pot size (normalized)

- **Targets**: `[batch_size, 2 * num_buckets]`
  - First `num_buckets` values: Player's counterfactual values
  - Next `num_buckets` values: Opponent's counterfactual values

- **Mask**: `[batch_size, num_buckets]`
  - Binary mask for valid actions (1 = valid, 0 = invalid)

## Network Architecture

The DeepStack value network uses the following architecture:

```
Input Layer: 2 * num_buckets + 1
    ↓
Hidden Layer 1: 256 neurons (PReLU activation)
    ↓
Hidden Layer 2: 256 neurons (PReLU activation)
    ↓
Hidden Layer 3: 128 neurons (PReLU activation)
    ↓
Output Layer: 2 * num_buckets
```

### Features

- **PReLU Activation**: Parametric ReLU for better gradient flow
- **Masked Huber Loss**: Robust loss function that handles invalid actions
- **Gradient Clipping**: Prevents exploding gradients (max_norm=1.0)
- **Early Stopping**: Stops training if validation loss doesn't improve for 10 epochs
- **Learning Rate Scheduling**: Reduces LR by 0.5 if validation loss plateaus for 3 epochs

## Training Process

### Phase 1: Initialization

1. Load training and validation data
2. Initialize neural network with specified architecture
3. Set up optimizer (Adam) and learning rate scheduler
4. Load existing best model if available (for continual training)

### Phase 2: Training Loop

For each epoch:
1. **Training Phase**:
   - Shuffle training data
   - Forward pass through network
   - Calculate masked Huber loss
   - Backward pass and gradient clipping
   - Update weights with Adam optimizer

2. **Validation Phase**:
   - Evaluate on validation set (no gradients)
   - Calculate validation loss

3. **Model Selection**:
   - If validation loss improved, save as best model
   - If no improvement for 10 epochs, trigger early stopping

4. **Learning Rate Adjustment**:
   - If validation loss plateaus for 3 epochs, reduce LR by 50%

### Phase 3: Model Saving

- **Best Model**: `models/pretrained/best_model.pt` (lowest validation loss)
- **Final Model**: `models/pretrained/final_model.pt` (last epoch's best)
- **Checkpoints**: `models/pretrained/checkpoint_epoch_N.pt` (every 10 epochs)

## Expected Results

### Training Progress

```
Epoch   1/50 | Train Loss: 0.297151 | Val Loss: 0.303743 | LR: 0.001000
Epoch   2/50 | Train Loss: 0.273254 | Val Loss: 0.283687 | LR: 0.001000
Epoch   3/50 | Train Loss: 0.251450 | Val Loss: 0.250361 | LR: 0.001000
Epoch   4/50 | Train Loss: 0.204057 | Val Loss: 0.205317 | LR: 0.001000
Epoch   5/50 | Train Loss: 0.159964 | Val Loss: 0.171609 | LR: 0.001000
...
```

### Performance Metrics

- **Training Time**: ~1-2 seconds per epoch on CPU, ~0.5 seconds on GPU
- **Convergence**: Typically converges in 20-40 epochs
- **Final Loss**: Target validation loss < 0.15 for good performance

## Using Trained Models

### Loading a Trained Model

```python
from deepstack.core.value_nn import ValueNN

# Load trained model
model = ValueNN(model_path='models/pretrained/best_model.pt', 
                num_hands=36, 
                device='cuda')

# Get value estimates
player_range = np.ones(36) / 36
opponent_range = np.ones(36) / 36
pot_size = 100.0

player_values, opponent_values = model.get_value_single(
    player_range, opponent_range, pot_size
)
```

### Integration with DeepStack

```python
from deepstack.core.resolving import Resolving

# Initialize resolver with trained model
resolver = Resolving(
    num_hands=36,
    game_variant='holdem',
    value_net_path='models/pretrained/best_model.pt'
)

# Use in continual re-solving
node_params = {
    'street': 0,
    'bets': [20, 20],
    'current_player': 1,
    'board': [],
    'bet_sizing': [1.0, 2.0]
}

resolver.resolve_first_node(node_params, player_range, opponent_range, 
                             iterations=1000)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Error (GPU)**
   - Solution: Reduce `batch_size` or use `--no-gpu`
   
2. **Data Loading Error**
   - Solution: Verify data files exist and have correct format
   - Check that `num_buckets` matches data dimensions

3. **Loss Not Decreasing**
   - Solution: Reduce learning rate (`--lr 0.0001`)
   - Try different architecture (`hidden_sizes`)

4. **Training Too Slow**
   - Solution: Use GPU (`--use-gpu`)
   - Increase `batch_size` (if memory allows)

### Debug Mode

```bash
# Print detailed configuration
python scripts/train_deepstack.py --verbose

# Test data loading only
python -c "
import sys
sys.path.insert(0, 'src')
from deepstack.core.data_stream import DataStream
ds = DataStream('data/deepstacked_training/samples/train_samples', 32, False)
print(f'Training samples: {ds.train_data_count}')
print(f'Validation samples: {ds.valid_data_count}')
"
```

## Advanced Topics

### Hyperparameter Tuning

Use Optuna for automated hyperparameter search:

```bash
python scripts/tune_hyperparams.py --target deepstack --trials 100
```

### Distributed Training

For faster training on multiple GPUs:

```python
# TODO: Implement multi-GPU training with DataParallel
```

### Model Optimization

After training, optimize the model for deployment:

```bash
# Prune and quantize
python scripts/optimize_model.py --model models/pretrained/best_model.pt
```

## References

- Original DeepStack Paper: [DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker](https://www.deepstack.ai/)
- Training Data Generation: See `data/deepstacked_training/Source/Training/`
- Network Architecture: Based on original Lua implementation

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `logs/` directory
3. Open an issue on GitHub with training configuration and error logs
