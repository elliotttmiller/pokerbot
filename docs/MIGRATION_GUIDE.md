# PokerBot Agent Migration Guide

## Overview

The pokerbot repository has consolidated multiple agent implementations into a single, unified, world-class agent called **PokerBotAgent**. This migration guide explains the changes and how to use the new agent.

## What Changed?

### Legacy Agents (Deprecated)
- `champion_agent.py` - ChampionAgent
- `elite_unified_agent.py` - EliteUnifiedAgent

These agents are now **deprecated** but remain in the codebase for backward compatibility. They will be maintained but not receive new features.

### New Unified Agent
- `pokerbot_agent.py` - **PokerBotAgent** (recommended for all new projects)

## Why the Change?

The consolidation was done to:
1. **Reduce Code Duplication**: ChampionAgent and EliteUnifiedAgent had significant overlap
2. **Improve Maintainability**: One agent is easier to maintain and test than multiple
3. **Modular Architecture**: All features are now configurable components
4. **Cleaner Codebase**: Simplified training pipeline and clearer structure
5. **Better Performance**: Optimized ensemble decision-making

## Features

PokerBotAgent combines all the best features from both legacy agents:

### From ChampionAgent:
- CFR (Counterfactual Regret Minimization)
- DQN (Deep Q-Network) 
- Pre-trained DeepStack models
- Preflop equity tables
- Enhanced neural network architecture

### From EliteUnifiedAgent:
- CFR+ with regret matching+
- DeepStack continual re-solving
- Action pruning and optimization
- Opponent modeling
- Monte Carlo simulation
- Card abstraction/bucketing

### New Capabilities:
- Fully modular component system
- Configurable ensemble weights
- Better lazy loading (faster startup)
- Improved training hooks
- Comprehensive test suite

## Usage

### Creating an Agent

#### New Way (Recommended):
```python
from agents import create_agent

# Create with default settings
agent = create_agent('pokerbot', name='MyBot')

# Create with custom configuration
agent = create_agent('pokerbot', 
                    name='CustomBot',
                    use_cfr=True,
                    use_cfr_plus=True,
                    use_dqn=True,
                    use_deepstack=True,
                    use_opponent_modeling=True,
                    use_pretrained=True,
                    cfr_weight=0.4,
                    dqn_weight=0.3,
                    deepstack_weight=0.3)

# Create with minimal features
agent = create_agent('pokerbot',
                    name='MinimalBot',
                    use_cfr=True,
                    use_dqn=False,
                    use_deepstack=False,
                    use_pretrained=False)
```

#### Old Way (Still Works):
```python
from agents import ChampionAgent, EliteUnifiedAgent

# These still work but are deprecated
champion = ChampionAgent(name='OldChamp')
elite = EliteUnifiedAgent(name='OldElite')
```

### Training

#### New Way:
```bash
# Train with PokerBot agent
python scripts/train.py --agent-type pokerbot --mode smoketest

# Train with full configuration
python scripts/train.py --agent-type pokerbot --mode standard --verbose
```

#### Old Way (Still Works):
```bash
# Train with Champion agent (default)
python scripts/train.py --mode smoketest
python scripts/train_champion.py --mode smoketest  # Backward compatible
```

### Configuration Options

```python
agent = create_agent('pokerbot',
    # Basic settings
    name='MyBot',                    # Agent name
    
    # Component toggles
    use_cfr=True,                    # Enable CFR component
    use_cfr_plus=True,               # Enable CFR+ enhancements
    use_dqn=True,                    # Enable DQN component
    use_deepstack=True,              # Enable DeepStack component
    use_opponent_modeling=True,      # Enable opponent modeling
    use_pretrained=True,             # Load pre-trained models
    
    # CFR configuration
    cfr_iterations=1000,             # CFR training iterations
    cfr_skip_iterations=500,         # Warmup iterations
    enable_pruning=True,             # Enable action pruning
    pruning_threshold=0.01,          # Pruning threshold
    
    # DQN configuration
    state_size=120,                  # State vector size
    action_size=5,                   # Number of actions
    learning_rate=0.0001,            # Learning rate
    gamma=0.99,                      # Discount factor
    epsilon=0.1,                     # Exploration rate
    epsilon_min=0.01,                # Minimum exploration
    epsilon_decay=0.9995,            # Exploration decay
    memory_size=50000,               # Replay buffer size
    
    # DeepStack configuration
    lookahead_depth=3,               # Lookahead depth
    
    # Ensemble weights (will be normalized)
    cfr_weight=0.4,                  # CFR decision weight
    dqn_weight=0.3,                  # DQN decision weight
    deepstack_weight=0.3             # DeepStack decision weight
)
```

## Migration Steps

### For Existing Projects

1. **Replace Agent Creation**:
   ```python
   # Old
   from agents import ChampionAgent
   agent = ChampionAgent(name='MyAgent')
   
   # New
   from agents import create_agent
   agent = create_agent('pokerbot', name='MyAgent')
   ```

2. **Update Training Scripts**:
   ```bash
   # Old
   python scripts/train_champion.py --mode smoketest
   
   # New
   python scripts/train.py --agent-type pokerbot --mode smoketest
   ```

3. **Update Imports**:
   ```python
   # Old
   from src.agents import ChampionAgent, EliteUnifiedAgent
   
   # New
   from agents import create_agent
   # or
   from src.agents.pokerbot_agent import PokerBotAgent
   ```

4. **Test Thoroughly**:
   ```bash
   python examples/test_pokerbot.py
   ```

### For New Projects

Simply use:
```python
from agents import create_agent
agent = create_agent('pokerbot', name='MyBot')
```

## API Compatibility

PokerBotAgent implements the same interface as the legacy agents:

```python
# All these methods work the same
agent.choose_action(hole_cards, community_cards, pot, current_bet, player_stack, opponent_bet)
agent.reset()
agent.observe_result(won, amount)
agent.train_cfr(num_iterations)
agent.remember(state, action, reward, next_state, done)
agent.replay(batch_size)
agent.save_models(directory)
agent.load_models(directory)
agent.set_training_mode(training)
agent.get_stats()
```

## Testing

A comprehensive test suite is available:

```bash
# Run all PokerBot tests
python examples/test_pokerbot.py

# Expected output:
# ============================================================
# PokerBot Agent Test Suite
# ============================================================
# 
# Testing PokerBot registry...
# Testing PokerBot agent creation...
# Testing PokerBot decision making...
# Testing PokerBot training...
# Testing PokerBot save/load...
# 
# ============================================================
# Test Results: 5/5 passed
# ============================================================
```

## Benefits of Migration

1. **Single Source of Truth**: One agent with all features
2. **Better Tested**: Comprehensive test suite
3. **More Maintainable**: Modular, well-documented code
4. **Easier Configuration**: All features are toggle-able
5. **Backward Compatible**: Old code still works
6. **Future-Proof**: New features will go into PokerBot only

## Timeline

- **Now**: PokerBot available, legacy agents deprecated
- **Future**: Legacy agents will remain for compatibility
- **Recommended**: Start using PokerBot for all new development

## Support

For questions or issues:
1. Check `examples/test_pokerbot.py` for usage examples
2. Review `src/agents/pokerbot_agent.py` source code
3. Run tests to verify everything works
4. Legacy agents remain available for backward compatibility

## Summary

The migration to PokerBotAgent provides a cleaner, more maintainable codebase while preserving all functionality. The new agent is:
- ✅ Feature-complete (all features from both legacy agents)
- ✅ Well-tested (comprehensive test suite)
- ✅ Modular and configurable
- ✅ Backward compatible (old agents still work)
- ✅ Recommended for all new development

Start using `create_agent('pokerbot')` today!
