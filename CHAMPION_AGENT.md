# Champion Agent - Unified CFR + DQN Poker AI

## Overview

The **Champion Agent** is our most advanced poker AI, unifying the best strategies from multiple approaches:

- **CFR (Counterfactual Regret Minimization)**: Game-theoretic optimal play
- **DQN (Deep Q-Network)**: Learned patterns through reinforcement learning
- **Pre-trained Models**: DeepStack champion neural networks (50,000+ epochs)
- **Equity Tables**: Preflop equity calculations for 169 starting hands

This creates a **superior, championship-level agent** that starts with advanced pre-trained knowledge instead of from scratch.

## Architecture

### Ensemble Decision Making

The Champion Agent uses a weighted voting system to combine strategies:

```python
champion = ChampionAgent(
    cfr_weight=0.4,      # 40% CFR strategy
    dqn_weight=0.4,      # 40% DQN neural network
    equity_weight=0.2    # 20% equity-based decisions
)
```

Each component "votes" on actions, and the final decision is made through weighted ensemble voting.

### Components

#### 1. CFR Component
- Game-theoretic optimal strategy through regret minimization
- Builds information sets for different game situations
- Converges to Nash equilibrium through self-play
- Best for: Exploiting predictable opponents, balanced play

#### 2. DQN Component
- Neural network learns patterns through experience
- Deep Q-learning with experience replay
- Adapts to opponent tendencies
- Best for: Exploitative play, adapting to specific opponents

#### 3. Pre-trained Models
- **DeepStack Models**: Champion-level neural networks
  - Located in `models/pretrained/`
  - 50,000+ epochs of training
  - ~1 MB each (CPU and GPU versions)
- **Equity Tables**: Preflop hand strength calculations
  - Located in `data/equity_tables/`
  - 169 starting hand combinations
  - Proven equity values

#### 4. Equity-Based Decisions
- Uses preflop equity tables for early-game decisions
- Makes informed decisions based on hand strength
- Premium hands (>65% equity): Raise aggressively
- Good hands (55-65% equity): Raise or call
- Playable hands (45-55% equity): Call reasonable bets
- Weak hands (<45% equity): Fold or check

## Quick Start

### Basic Usage

```python
from src.agents import ChampionAgent

# Create champion agent with pre-trained models
champion = ChampionAgent(
    name="Champion",
    use_pretrained=True
)

# Make a decision
action, raise_amount = champion.choose_action(
    hole_cards=[Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)],
    community_cards=[],
    pot=100,
    current_bet=20,
    player_stack=1000,
    opponent_bet=20
)

print(f"Decision: {action.name}, Raise: ${raise_amount}")
```

### Custom Configuration

```python
# Configure ensemble weights
champion = ChampionAgent(
    name="CustomChampion",
    use_pretrained=True,
    cfr_weight=0.5,      # More weight on CFR
    dqn_weight=0.3,      # Less weight on DQN
    equity_weight=0.2,   # Same equity weight
    epsilon=0.1          # Low exploration (more exploitation)
)
```

### Training

The Champion Agent supports two training modes:

#### 1. CFR Training (Self-Play)

```python
# Train CFR component through self-play
champion.train_cfr(num_iterations=10000)
```

#### 2. DQN Training (Reinforcement Learning)

```python
from src.evaluation import Trainer
from src.agents import RandomAgent

# Train against an opponent
opponent = RandomAgent()
trainer = Trainer(champion, opponent)

trainer.train(
    num_episodes=1000,
    batch_size=32,
    save_interval=100
)
```

### Saving and Loading

```python
# Save trained strategy
champion.save_strategy("models/my_champion")
# Creates: my_champion.cfr (CFR strategy)
#          my_champion.dqn (DQN model)

# Load trained strategy
champion2 = ChampionAgent(name="Champion2", use_pretrained=True)
champion2.load_strategy("models/my_champion")
```

## Pre-trained Model Integration

### How Pre-trained Models Are Integrated

The Champion Agent automatically loads pre-trained models on initialization:

1. **DeepStack Models**: Neural networks trained on 50K+ epochs
   - Used for value estimation and decision refinement
   - Available in CPU and GPU versions
   - From the DeepStack-Leduc project

2. **Equity Tables**: Preflop hand strength calculations
   - 169 starting hand combinations
   - Proven equity values from simulation
   - Used for informed preflop decisions

### Checking Pre-trained Model Status

```python
champion = ChampionAgent(name="Test", use_pretrained=True)

# Check if models loaded
if champion.model_loader:
    models = champion.model_loader.list_available_models()
    print(f"Loaded {len(models)} pre-trained models")

if champion.data_manager:
    stats = champion.data_manager.get_dataset_stats()
    print(f"Equity tables: {stats['preflop_hands']} hands")
```

## Advantages Over Individual Agents

### vs. Pure CFR Agent
- ✅ Better adaptability through DQN learning
- ✅ Starts with pre-trained knowledge (not from scratch)
- ✅ More nuanced decision-making through ensemble voting
- ✅ Equity-based preflop optimization

### vs. Pure DQN Agent
- ✅ Game-theoretic foundation from CFR
- ✅ More stable in unexplored situations
- ✅ Better exploitability-resistance
- ✅ Preflop decisions backed by proven equity

### vs. Both Separately
- ✅ Unified strategy combining both strengths
- ✅ Single agent interface
- ✅ Coordinated training of both components
- ✅ Optimal weight balancing through ensemble

## Performance Characteristics

### Strengths
- **Champion-level baseline**: Starts with 50K+ epochs of pre-trained knowledge
- **Balanced play**: Combines exploitative (DQN) and equilibrium (CFR) strategies
- **Adaptability**: Learns from experience while maintaining theoretical foundation
- **Informed decisions**: Uses equity tables for optimal preflop play

### When to Use Champion Agent
- ✅ High-stakes games requiring optimal play
- ✅ Tournaments with diverse opponents
- ✅ When you want the best overall agent
- ✅ Starting a new training regime (better baseline)

### Training Time Savings
Starting with pre-trained models provides:
- **~50,000 epochs** of neural network training
- **Proven equity calculations** for all starting hands
- **Game-theoretic strategies** from CFR foundation
- Saves **weeks or months** of training time

## API Reference

### Constructor

```python
ChampionAgent(
    name: str = "ChampionAgent",
    state_size: int = 60,
    action_size: int = 3,
    learning_rate: float = 0.001,
    gamma: float = 0.95,
    epsilon: float = 0.3,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.995,
    memory_size: int = 2000,
    use_pretrained: bool = True,
    cfr_weight: float = 0.4,
    dqn_weight: float = 0.4,
    equity_weight: float = 0.2
)
```

### Methods

#### `choose_action(hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet)`
Choose action using ensemble of CFR, DQN, and equity strategies.

**Returns:** `(Action, raise_amount)`

#### `train_cfr(num_iterations)`
Train CFR component through self-play.

#### `remember(state, action, reward, next_state, done)`
Store experience in DQN replay memory.

#### `replay(batch_size=32)`
Train DQN model on batch of experiences.

#### `save_strategy(filepath_prefix)`
Save both CFR and DQN strategies to files.

#### `load_strategy(filepath_prefix)`
Load both CFR and DQN strategies from files.

#### `set_training_mode(training: bool)`
Enable or disable training mode (affects exploration).

## Examples

### Example 1: Running the Demo

```bash
python demo_champion.py
```

This demonstrates:
- Champion agent initialization with pre-trained models
- Decision-making in various scenarios
- Comparison with other agents
- Saving and loading strategies

### Example 2: Evaluating Champion Agent

```bash
python evaluate.py --num-hands 1000 --agents champion random --verbose
```

### Example 3: Custom Training Pipeline

```python
from src.agents import ChampionAgent, RandomAgent
from src.evaluation import Trainer

# Create champion agent
champion = ChampionAgent(
    name="MyChampion",
    use_pretrained=True,
    cfr_weight=0.5,
    dqn_weight=0.3,
    equity_weight=0.2
)

# First: Train CFR component
print("Training CFR component...")
champion.train_cfr(num_iterations=5000)

# Second: Train DQN component against opponents
print("Training DQN component...")
opponent = RandomAgent()
trainer = Trainer(champion, opponent)
trainer.train(num_episodes=500, batch_size=32)

# Save the trained agent
champion.save_strategy("models/my_trained_champion")

print("Training complete!")
```

## Technical Details

### State Encoding
The Champion Agent encodes game state as a 60-dimensional vector:
- Hole cards (4 dims: 2 cards × 2 features)
- Community cards (10 dims: 5 cards × 2 features)
- Game state (5 dims: pot, current_bet, stack, opponent_bet, round)
- Equity info (1 dim: preflop equity if available)
- Additional features (40 dims: reserved for future enhancements)

### Ensemble Voting Algorithm
1. Each component (CFR, DQN, Equity) casts a vote for an action
2. Votes are weighted by configured weights (normalized to sum to 1.0)
3. Action with highest weighted score is selected
4. For RAISE actions, median of suggested amounts is used

### Hand Notation Format
The equity table uses specific notation:
- Pairs: `AA`, `KK`, `QQ`, etc.
- Non-pairs: Lower rank first: `KAS` (not `AKS`), `27O` (not `72O`)
- `S` = suited, `O` = offsuit

## Troubleshooting

### Pre-trained Models Not Loading

**Problem:** "DeepStack model not found" or "Equity tables not found"

**Solution:** Ensure models are in the correct directories:
- DeepStack models: `models/pretrained/final_cpu.model`
- Equity tables: `data/equity_tables/preflop_equity.json`

### TensorFlow Not Available

**Message:** "TensorFlow not available. Champion agent will use CFR + equity only."

**Impact:** DQN component won't function, but CFR and equity-based decisions still work.

**Solution:** Install TensorFlow:
```bash
pip install tensorflow>=2.16.0
```

### Memory Issues During Training

**Problem:** Out of memory during DQN training

**Solution:** Reduce memory size:
```python
champion = ChampionAgent(memory_size=1000)  # Default is 2000
```

## Future Enhancements

Potential improvements for the Champion Agent:

- [ ] Multi-opponent strategies
- [ ] Range-based reasoning
- [ ] Bankroll management integration
- [ ] Tournament-specific strategies
- [ ] Online learning and adaptation
- [ ] GPU acceleration for DQN
- [ ] Integration with additional pre-trained models

## References

1. **CFR Algorithm**: Zinkevich et al., "Regret Minimization in Games with Incomplete Information"
2. **DQN**: Mnih et al., "Playing Atari with Deep Reinforcement Learning"
3. **DeepStack**: Moravčík et al., "DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker"
4. **DeepStack-Leduc**: https://github.com/lifrordi/DeepStack-Leduc

## License

This agent integrates pre-trained models from open-source projects. Please respect the original licenses.

## Contributing

To contribute improvements to the Champion Agent:

1. Test thoroughly with `test_champion.py`
2. Ensure backward compatibility
3. Document new features
4. Submit pull request with clear description

---

🏆 **The Champion Agent represents the pinnacle of our poker AI - combining proven strategies with championship-level pre-trained knowledge!**
