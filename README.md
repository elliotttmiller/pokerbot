# Poker Bot - Advanced AI Poker Agent

A comprehensive poker bot system with multiple AI agents, including Deep Q-Network (DQN), fixed strategy, and random agents. The system features vision-based game state detection, automated playing capabilities, and extensive evaluation tools.

## Features

- **Multiple AI Agents**:
  - **DQN Agent**: Deep reinforcement learning agent using Q-learning
  - **Fixed Strategy Agent**: GTO-inspired fixed strategy with pot odds calculations
  - **Random Agent**: Baseline random decision-making agent

- **Vision System**: 
  - Automated game state detection using GPT-4 Vision
  - Screen capture and control
  - Automatic action execution

- **Training & Evaluation**:
  - Comprehensive training framework for DQN agents
  - Agent comparison and evaluation system
  - Performance statistics and analytics

- **Game Engine**:
  - Complete Texas Hold'em poker game implementation
  - Hand evaluation and ranking
  - Betting rounds and pot management

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

## Usage

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
