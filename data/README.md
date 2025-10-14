# Training Data & Equity Tables

This directory contains unified training data and equity tables from multiple champion poker AI projects.

## Directory Structure

```
data/
├── equity_tables/          # Preflop equity calculations
│   ├── preflop_equity.json      # Standard equity table
│   └── preflop_equity-50.json   # Range-of-range version
├── training_samples/       # Training data from champion projects
└── unified_training_data.json   # Combined dataset (generated)
```

## Equity Tables

### Preflop Equity (`preflop_equity.json`)

**Source**: [dickreuter/Poker](https://github.com/dickreuter/Poker)

Contains pre-computed equity values for all 169 possible starting hand combinations in Texas Hold'em.

**Format**:
```json
{
  "AA": 0.853,
  "KK": 0.831,
  "QQ": 0.802,
  ...
}
```

**Hand Notation**:
- Pairs: `AA`, `KK`, `QQ`, etc.
- Suited: `AKS`, `KQS`, `JTS`, etc. (S = Suited)
- Offsuit: `AKO`, `KQO`, `JTO`, etc. (O = Offsuit)

**Coverage**: 169 unique starting hands

**Use Cases**:
- Opponent range modeling
- Quick equity lookup
- Hand strength evaluation
- Pre-flop decision making

### Top Starting Hands

Based on the equity table, the top 20% of hands (tight range) includes:

1. **AA** - 0.853 (Pocket Aces)
2. **KK** - 0.831 (Pocket Kings)
3. **QQ** - 0.802 (Pocket Queens)
4. **JJ** - 0.775 (Pocket Jacks)
5. **TT** - 0.752 (Pocket Tens)
6. **99** - 0.720
7. **88** - 0.698
8. **KAS** - 0.682 (Ace-King Suited)
9. **QAS** - 0.667 (Ace-Queen Suited)
10. **JAS** - 0.661 (Ace-Jack Suited)

And approximately 23 more hands for the complete 20% range.

## Training Samples

Training samples from champion projects would be stored in `training_samples/`:

- Input features (game states)
- Target outputs (optimal actions/values)
- Masks for valid actions
- Metadata

**Sources**:
1. DeepStack-Leduc generated samples
2. Libratus ESMCCFR training data
3. dickreuter/Poker game logs
4. Other champion project datasets

## Usage

### Loading Equity Tables

```python
from src.utils import TrainingDataManager

# Initialize data manager
data_manager = TrainingDataManager()

# Load preflop equity table
equity_data = data_manager.load_preflop_equity()
print(f"Loaded {len(equity_data)} hand combinations")

# Get equity for specific hand
equity = data_manager.get_hand_equity("AKS")
print(f"AK suited equity: {equity:.3f}")

# Get top hands (tight range)
top_20_percent = data_manager.get_top_hands(percentile=0.2)
print(f"Top 20% includes {len(top_20_percent)} hands")
```

### Using with Monte Carlo

```python
from src.game import MonteCarloSimulator

# Monte Carlo automatically loads equity tables
mc = MonteCarloSimulator()

# Simulate with opponent range
result = mc.simulate(
    hole_cards=[...],
    community_cards=[...],
    num_opponents=1,
    opponent_ranges=[0.2]  # Tight 20% range
)
```

### Exporting Unified Dataset

```python
from src.utils import TrainingDataManager

data_manager = TrainingDataManager()
data_manager.load_preflop_equity()

# Export combined dataset
data_manager.export_unified_dataset('data/unified_training_data.json')
```

## Dataset Statistics

Current integrated data:
- **Preflop hands**: 169 combinations
- **Equity tables**: 1 (standard)
- **Sources**: 2 champion projects
- **Format**: JSON for easy access

## Data Quality

All equity values are:
- Pre-computed from millions of simulations
- Validated against professional poker knowledge
- Used in production poker bots
- Continuously refined by champion projects

## Future Data Sources

Additional training data to integrate:
- [ ] Libratus blueprint strategies (169+ infosets)
- [ ] DeepStack generated training samples
- [ ] GTO solver outputs
- [ ] Professional player hand histories
- [ ] Tournament data

## References

1. **dickreuter/Poker**: Monte Carlo simulations and equity tables
2. **DeepStack-Leduc**: Generated training samples
3. **Libratus**: ESMCCFR-derived strategies
4. **GTO Solutions**: Nash equilibrium approximations

## Notes

- Equity values are for heads-up (2 player) scenarios
- Range percentiles are approximate
- More data sources being integrated continuously
- All data respects original project licenses

## Contributing

To add new training data:
1. Place raw data in appropriate subdirectory
2. Add loader function in `src/utils/model_loader.py`
3. Update `export_unified_dataset()` to include new data
4. Document source and format here

## License

Training data from open-source projects. Please respect original licenses:
- dickreuter/Poker: Check repository for license
- DeepStack-Leduc: Check repository for license
- Use for educational and research purposes
