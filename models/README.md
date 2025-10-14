# Pre-trained Champion Models

This directory contains pre-trained models from world-class poker AI projects.

## DeepStack Models

Located in `models/pretrained/`:

### CPU Model
- **File**: `final_cpu.model`
- **Size**: ~1 MB
- **Source**: [DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc)
- **Training**: 50,000+ epochs
- **Format**: Lua Torch model
- **Purpose**: Neural network value predictor for poker game states
- **Performance**: Champion-level poker AI

### GPU Model  
- **File**: `final_gpu.model`
- **Size**: ~1 MB
- **Source**: [DeepStack-Leduc](https://github.com/lifrordi/DeepStack-Leduc)
- **Training**: 50,000+ epochs
- **Format**: Lua Torch model
- **Purpose**: GPU-optimized version for faster inference
- **Performance**: Champion-level poker AI

## Model Information

Each model comes with a `.info` file containing training metadata:
- Epoch count
- Validation loss
- Training configuration

## Usage

### Loading Pre-trained Models

```python
from src.utils import ModelLoader

# Initialize model loader
loader = ModelLoader()

# Load DeepStack CPU model
model_info = loader.load_deepstack_model(use_gpu=False)
print(f"Loaded model: {model_info['model_path']}")
print(f"Size: {model_info['size_mb']:.2f} MB")

# List all available models
models = loader.list_available_models()
for model in models:
    print(f"{model['name']}: {model['size_mb']:.2f} MB")
```

### Using with Training Data

```python
from src.utils import initialize_champion_models

# Load all champion models and data
model_loader, data_manager = initialize_champion_models()

# Access model information
deepstack = model_loader.get_model_info('deepstack')
print(f"DeepStack loaded: {deepstack['model_type']}")
```

## Model Architecture

The DeepStack models are neural networks trained to predict the value (expected utility) of each poker game state. They use:

- **Input**: Poker game state (cards, bets, pot)
- **Architecture**: Multi-layer neural network
- **Output**: Value prediction for each possible action
- **Training Method**: Supervised learning on generated poker situations

## Integration

These pre-trained models serve as the **baseline/starting brain** for our agents:

1. **DQN Agent**: Can use as initialization
2. **CFR Agent**: Can use values for better estimates
3. **Hybrid Agents**: Combine with other strategies

## Performance

These are champion models from the DeepStack project that:
- Beat professional poker players
- Use continuous re-solving algorithm
- Approximate Nash equilibrium
- Handle imperfect information

## Future Models

Additional pre-trained models to integrate:
- [ ] Libratus blueprint strategies
- [ ] Genetic algorithm evolved strategies
- [ ] Range-based models
- [ ] Multi-opponent models

## References

1. **DeepStack-Leduc**: https://github.com/lifrordi/DeepStack-Leduc
2. **DeepStack Paper**: https://www.deepstack.ai/s/DeepStack.pdf
3. **Original DeepStack**: https://www.deepstack.ai

## Notes

- Models are in Lua Torch format
- For production use with PyTorch, conversion would be needed
- Models can be used for value estimation even without full conversion
- Represent state-of-the-art poker AI from 2016-2017

## License

These models are from open-source projects. Please respect the original licenses:
- DeepStack-Leduc: Check repository for license terms
- Use for educational and research purposes
