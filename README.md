# Poker Bot - Advanced AI Poker Agent

A comprehensive poker bot system with multiple AI agents, including advanced CFR with pruning, DQN, and unified champion agents. The system features vision-based game state detection, distributed training, and real-time search capabilities.

## Features

### 🏆 Advanced AI Agents

- **Champion Agent**: Unified CFR + DQN hybrid with pre-trained models
- **Search Agent**: Real-time depth-limited search with blueprint strategy
- **Advanced CFR Agent**: CFR with pruning, linear discounting, and progressive training
- **DQN Agent**: Deep reinforcement learning agent using Q-learning
- **CFR Agent**: Counterfactual Regret Minimization for game-theoretic optimal play
- **Fixed Strategy Agent**: GTO-inspired fixed strategy with pot odds calculations
- **Random Agent**: Baseline random decision-making agent

### 🚀 Progressive Training Pipeline

- **Multi-phase CFR Training**: Warmup → Pruning → Linear CFR → Strategy updates
- **Distributed Training**: Multiprocessing for 4-8x speedup
- **Card Abstraction**: Information set clustering for memory efficiency
- **Vicarious Learning**: Multi-agent training and imitation learning

### 🎯 Advanced Features

- **Blueprint + Real-time Search**: Two-stage decision making
- **Information Set Abstraction**: Reduces 56B+ states to tractable sizes
- **CFR with Pruning (CFRp)**: 20x+ speedup through action pruning
- **Linear CFR**: Faster convergence via discounting
- **Pre-trained Models**: DeepStack (50K+ epochs) and equity tables

### 🎮 Vision System

- Automated game state detection using GPT-4 Vision
- Screen capture and control
- Automatic action execution

### 📊 Training & Evaluation

- Comprehensive training framework
- Distributed training with multiprocessing
- Agent comparison and evaluation system
- Performance statistics and analytics

## Project Structure

```
pokerbot/
├── src/                    # Source code
│   ├── agents/            # AI agent implementations
│   │   ├── champion_agent.py
│   │   ├── search_agent.py
│   │   ├── advanced_cfr.py
│   │   ├── cfr_agent.py
│   │   ├── dqn_agent.py
│   │   └── ...
│   ├── game/              # Game engine and utilities
│   │   ├── card_abstraction.py
│   │   ├── game_state.py
│   │   ├── hand_evaluator.py
│   │   └── ...
│   ├── evaluation/        # Training and evaluation
│   │   ├── distributed_trainer.py
│   │   ├── trainer.py
│   │   └── ...
│   └── utils/             # Utility modules
├── scripts/               # Executable scripts
│   ├── train.py
│   ├── play.py
│   └── evaluate.py
├── examples/              # Examples and demos
│   ├── demo_champion.py
│   ├── example_champion.py
│   └── test_champion.py
├── docs/                  # Documentation
│   ├── CHAMPION_AGENT.md
│   ├── PLURIBUS_ANALYSIS.md
│   └── ...
├── models/                # Saved models and pre-trained weights
├── data/                  # Training data and equity tables
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/elliotttmiller/pokerbot.git
cd pokerbot
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
export OPENAI_API_KEY=your_api_key_here
```
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional):
```bash
export OPENAI_API_KEY=your_api_key_here
```

## Usage

### 🏆 Using the Champion Agent

The Champion Agent is our most advanced AI, combining CFR, DQN, and pre-trained models:

```bash
# Run the demo to see it in action
python demo_champion.py
```

Quick start in code:
```python
from src.agents import ChampionAgent
from src.game import Card, Rank, Suit

# Create champion agent with pre-trained knowledge
champion = ChampionAgent(name="Champion", use_pretrained=True)

# Make a decision
hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)]
action, raise_amount = champion.choose_action(
    hole_cards=hole_cards,
    community_cards=[],
    pot=100,
    current_bet=20,
    player_stack=1000,
    opponent_bet=20
)
```

**See [CHAMPION_AGENT.md](CHAMPION_AGENT.md) for complete documentation.**

### Training a DQN Agent

Train a new DQN agent through self-play:

```bash
python train.py --episodes 1000 --batch-size 32 --verbose
```

Options:
- `--episodes`: Number of training episodes (default: 1000)
- `--batch-size`: Batch size for training (default: 32)
- `--save-interval`: Save model every N episodes (default: 100)
- `--model-dir`: Directory to save models (default: models)
- `--verbose`: Print training progress

### Evaluating Agents

Compare different agents against each other:

```bash
python evaluate.py --num-hands 1000 --agents dqn fixed random --verbose
```

Options:
- `--num-hands`: Number of hands to play (default: 1000)
- `--model-path`: Path to trained DQN model
- `--agents`: Agents to evaluate (choices: dqn, fixed, random)
- `--verbose`: Print evaluation progress

### Playing Automated Poker

Run the bot to play on an online poker site:

```bash
python play.py --agent fixed --max-hands 20 --delay 2.0 --verbose
```

Options:
- `--agent`: Agent type (choices: dqn, fixed)
- `--model-path`: Path to trained DQN model (for DQN agent)
- `--max-hands`: Maximum hands to play (default: 20)
- `--delay`: Delay between actions in seconds (default: 2.0)
- `--verbose`: Print detailed information

**Note**: Vision-based detection requires an OpenAI API key for GPT-4 Vision.

## Project Structure

```
pokerbot/
├── src/
│   ├── agents/              # AI agent implementations
│   │   ├── base_agent.py    # Base agent interface
│   │   ├── dqn_agent.py     # Deep Q-Network agent
│   │   ├── fixed_strategy_agent.py  # Fixed strategy agent
│   │   └── random_agent.py  # Random agent
│   ├── game/                # Game engine
│   │   ├── card.py          # Card and deck classes
│   │   ├── hand_evaluator.py  # Hand ranking evaluation
│   │   └── game_state.py    # Game state management
│   ├── vision/              # Vision and automation
│   │   ├── vision_detector.py  # Game state detection
│   │   └── screen_controller.py  # Screen control
│   ├── evaluation/          # Training and evaluation
│   │   ├── trainer.py       # Training framework
│   │   └── evaluator.py     # Evaluation framework
│   └── utils/               # Utilities
│       ├── config.py        # Configuration management
│       └── logger.py        # Logging utilities
├── data/                    # Data files
├── models/                  # Trained models
├── screenshots/             # Game screenshots
├── logs/                    # Log files
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── play.py                  # Automated playing script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Architecture

### Game Engine

The game engine provides a complete implementation of Texas Hold'em poker:
- Card representation with ranks and suits
- Deck management with shuffling
- Hand evaluation using standard poker rankings
- Game state management with betting rounds
- Player actions (fold, check, call, raise)

### AI Agents

#### DQN Agent
- Uses deep neural networks to learn optimal strategy
- Experience replay for stable training
- Epsilon-greedy exploration strategy
- State encoding with normalized features

#### Fixed Strategy Agent
- Implements GTO-inspired strategy
- Pot odds calculations
- Hand strength evaluation
- Pre-flop and post-flop strategies

#### Random Agent
- Baseline for comparison
- Makes random valid decisions

### Vision System

- **VisionDetector**: Uses GPT-4 Vision to extract game state from screenshots
- **ScreenController**: Captures screenshots and controls mouse/keyboard
- **ActionMapper**: Maps poker actions to screen coordinates

## Configuration

Configuration can be set via environment variables:

```bash
# Game settings
export STARTING_STACK=1000
export SMALL_BLIND=10
export BIG_BLIND=20

# Training settings
export NUM_EPISODES=1000
export BATCH_SIZE=32

# Vision settings
export OPENAI_API_KEY=your_key_here
export SCREENSHOT_DIR=screenshots

# Model settings
export MODEL_DIR=models
export MODEL_PATH=models/model_final.h5

# Logging
export VERBOSE=true
export LOG_DIR=logs
```

## Performance

Example results after 1000 hands:

| Agent | Wins | Win Rate | Total Winnings |
|-------|------|----------|----------------|
| Fixed Strategy | 927 | 92.7% | +$18,540 |
| DQN Agent | 51 | 5.1% | -$9,270 |
| Random Agent | 6 | 0.6% | -$9,270 |
| Ties | 16 | 1.6% | - |

*Note: DQN performance improves significantly after training.*

## References

This project incorporates concepts and techniques from:

1. **GTO Poker Bot** - https://github.com/arnenoori/gto-poker-bot
   - Vision-based game state detection
   - Automated playing framework

2. **Poker AI** - https://github.com/dickreuter/Poker
   - Advanced poker strategies
   - Pre-trained models

3. **DeepStack-Leduc** - https://github.com/lifrordi/DeepStack-Leduc
   - Deep counterfactual regret minimization

4. **Libratus** - https://noambrown.github.io/papers/17-IJCAI-Libratus.pdf
   - Game theory optimal strategies
   - Abstraction techniques

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is for educational and research purposes only. Please ensure compliance with all applicable laws and terms of service when using this software.
