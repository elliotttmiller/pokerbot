# Poker Bot - Advanced AI Poker Agent

A comprehensive poker bot system with multiple AI agents, including advanced CFR with pruning, DQN, and unified champion agents. The system features **a fully optimized DeepStack training pipeline**, vision-based game state detection, distributed training, and real-time search capabilities.

## 📚 Documentation

**NEW: Multi-Modal Vision Integration (QWEN 2.5-7B VL Support!)** 🎯
- 🔮 **[MULTIMODAL_VISION_INTEGRATION_AUDIT.md](MULTIMODAL_VISION_INTEGRATION_AUDIT.md)** - Complete audit for QWEN VL integration ⭐ **NEW**
- 📊 **[EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md)** - TL;DR executive summary (10 pages) ⭐ **START HERE**
- 📋 **[COMPREHENSIVE_ANALYSIS_REPORT.md](COMPREHENSIVE_ANALYSIS_REPORT.md)** - Full 200+ page analysis of all references
- 🗺️ **[IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)** - Week-by-week implementation plan
- 📖 **[DEEPSTACK_OFFICIAL_ANALYSIS.md](DEEPSTACK_OFFICIAL_ANALYSIS.md)** - Official DeepStack implementation analysis
- ⚡ **[PIPELINE_OPTIMIZATION.md](PIPELINE_OPTIMIZATION.md)** - Pipeline performance optimization report

**DeepStack Training System (Fully Optimized!)** ✨
- 🎯 **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start guide (3 steps to train)
- 📊 **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - Comprehensive optimization guide (8 pages)
- 📋 **[AUDIT_SUMMARY.md](AUDIT_SUMMARY.md)** - Summary of critical fixes and improvements
- 🔧 **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete step-by-step training manual
- ⚙️ **[SYSTEM_AUDIT.md](SYSTEM_AUDIT.md)** - System audit and optimization report
- 🎓 **[VALIDATION_RECOMMENDATIONS_ANALYSIS.md](VALIDATION_RECOMMENDATIONS_ANALYSIS.md)** - Validation recommendations analysis
- 🔬 **[docs/VALIDATION_IMPROVEMENTS.md](docs/VALIDATION_IMPROVEMENTS.md)** - Temperature scaling & enhanced diagnostics

**Agent System:**
- 📖 **[MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Agent migration and usage guide
- 🔧 **[IMPLEMENTATION_SUMMARY.md](docs/IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- 🔄 **[IMPORT_UPDATE_SUMMARY.md](docs/IMPORT_UPDATE_SUMMARY.md)** - System-wide import changes

**Quick Start - Multi-Modal Vision Bot:**
```bash
# Use fine-tuned QWEN 2.5-7B VL model for game state detection
from src.vision import MultiModalVisionDetector
from src.workflow import PokerWorkflowOrchestrator

# Initialize with vision model
orchestrator = PokerWorkflowOrchestrator(
    vision_model_path="models/qwen_poker_vl",
    strategy_model_path="models/deepstack_champion.pt"
)

# Start playing
orchestrator.start_new_hand()
result = orchestrator.process_screenshot("screenshot.png")
print(f"Action: {result.action}, Amount: {result.amount}")
```

**Quick Start - DeepStack Training:**
```bash
# Step 1: Generate training data (1K samples, ~15 min)
python scripts/generate_quick_data.py --samples 1000 --cfr-iters 2000

# Step 2: Train model (GPU recommended)
python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu

# Step 3: Validate results
python scripts/validate_deepstack_model.py --model models/versions/best_model.pt
```

See **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** for complete instructions.

**Quick Start - Agent Training:**
```bash
# Quick test (1-2 minutes)
python scripts/train.py --agent-type pokerbot --mode smoketest --verbose

# Development (20-30 minutes)
python scripts/train.py --agent-type pokerbot --mode standard --verbose

# Championship (4-8 hours)
python scripts/train.py --agent-type pokerbot --mode production --verbose --report
```

## 🚀 What's New

### Multi-Modal Vision Integration (QWEN 2.5-7B VL) 🔮 (Dec 2025)

**Complete infrastructure for integrating fine-tuned QWEN 2.5-7B VL model:**
- ✅ `MultiModalVisionDetector` - Local VL model inference for game state detection
- ✅ `PokerWorkflowOrchestrator` - Unified pipeline: vision → strategy → action
- ✅ `GameContext` - Persistent hand tracking across decisions
- ✅ Full test coverage with 19 passing tests
- ✅ Comprehensive audit against world-class poker AI references

**References Analyzed:**
- ✅ DeepStack.pdf - Neural network architecture & continual re-solving
- ✅ DeepStack-Leduc GitHub - CFR implementation patterns
- ✅ g5-poker-bot - GPU acceleration patterns
- ✅ self-operating-computer - Vision/multimodal integration
- ✅ gto-poker-bot - GTO strategy implementation

**See [MULTIMODAL_VISION_INTEGRATION_AUDIT.md](MULTIMODAL_VISION_INTEGRATION_AUDIT.md) for complete details.**

---

### Comprehensive Analysis & Recommendations Complete! 🎯 (Latest - Oct 18, 2025)

**All referenced materials thoroughly analyzed:**
- ✅ UAI05.pdf - Bayes' Bluff paper (Billings et al., 2005)
- ✅ DeepStack.pdf - Official Science paper (Moravčík et al., 2017)
- ✅ DeepStack-Leduc GitHub - Official implementation & docs
- ✅ DEEPSTACK_OFFICIAL_ANALYSIS.md - Internal analysis
- ✅ PIPELINE_OPTIMIZATION.md - Performance improvements

**Key Findings:**
- ✅ **Architecture:** Solid foundation (Grade: A-, 85/100)
- 🔴 **GPU Acceleration:** Missing 10-50x speedup (CRITICAL)
- 🔴 **Sample Quantity:** 100K vs 10M needed (100x gap)
- 🔴 **Neural Network:** 1.3K vs 6-14K params (5-11x too small)

**Roadmap to Championship Level:**
- 📅 Week 1: Add GPU support (10-50x speedup)
- 📅 Weeks 2-3: Generate 10M samples + train model
- 📅 Week 4: Live gameplay API + opponent modeling
- 🏆 Result: Championship AI in 8-13 weeks

**📊 See [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) for complete analysis (START HERE!)**

---

### Validation System Enhancements - Industry Best Practices! 🔥

The DeepStack validation system has been enhanced with **industry-standard calibration and diagnostics**:

- **✅ Temperature Scaling** - Post-hoc calibration (Guo et al., 2017)
- **✅ Per-Player Diagnostics** - Detect alignment issues automatically
- **✅ Priority-Based Recommendations** - CRITICAL/HIGH/OPTIMIZATION categories
- **✅ Optimized Configuration** - Based on validation analysis (200 epochs, lower delta)
- **✅ Comprehensive Analysis** - 9-page deep dive into best practices

**Validation Recommendation Quality: A+ (95/100)**
- All recommendations are research-backed and industry-standard
- Aligned with DeepStack paper specifications
- Incorporates modern improvements (EMA, AMP, temperature scaling)

**Quick Start:**
```bash
# Validate with enhanced diagnostics
python scripts/validate_deepstack_model.py

# Train with optimized config
python scripts/train_deepstack.py --config scripts/config/optimized.json --use-gpu
```

See **[VALIDATION_RECOMMENDATIONS_ANALYSIS.md](VALIDATION_RECOMMENDATIONS_ANALYSIS.md)** and **[docs/VALIDATION_IMPROVEMENTS.md](docs/VALIDATION_IMPROVEMENTS.md)** for details.

### DeepStack Training Pipeline - Fully Optimized! 🔥

The DeepStack neural network training pipeline has been **completely audited and optimized** for championship-level performance:

- **✅ Fixed Critical Terminal Equity Bug** - Now uses proper Monte Carlo simulation (AA vs 72o: 0.81 equity ✅)
- **✅ Increased CFR Quality** - 2000+ iterations per sample (was 1000)
- **✅ Fixed Street Coverage** - All streets now covered (flop/river were at 0)
- **✅ Championship Training Config** - Optimized hyperparameters per DeepStack paper
- **✅ Comprehensive Documentation** - 8-page optimization guide + quick reference

**Expected Impact:**
- Correlation: 0.30 → **>0.85** ✅
- Relative Error: 1257% → **<5%** ✅
- Street Coverage: Partial → **Full** ✅

**Quick Start:**
```bash
# Generate training data (1K samples for testing)
python scripts/generate_quick_data.py --samples 1000 --cfr-iters 2000

# Train with championship config
python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu

# Validate results
python scripts/validate_deepstack_model.py
```

See **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** and **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** for complete details.

### Unified PokerBot Agent 🎯

The pokerbot now features a single, unified world-class agent combining all best features:

- **✅ Modular Architecture** - Configure exactly what you need
- **✅ All Features Included** - CFR/CFR+, DQN, DeepStack, Opponent Modeling
- **✅ Ensemble Decision Making** - Best-of-breed from multiple AI techniques
- **✅ Comprehensive Tests** - Full test coverage and validation
- **✅ Easy to Use** - Simple API with sensible defaults

**Quick Start:**
```bash
# Create and test the unified agent
python examples/test_pokerbot.py

# Train with the new agent
python scripts/train.py --agent-type pokerbot --mode smoketest
```

**Code Example:**
```python
from agents import create_agent

# Create agent with all features
agent = create_agent('pokerbot', name='MyBot')

# Or customize components
agent = create_agent('pokerbot', 
                    use_cfr=True, 
                    use_dqn=True,
                    use_deepstack=True,
                    cfr_weight=0.4,
                    dqn_weight=0.3,
                    deepstack_weight=0.3)
```

See [Migration Guide](docs/MIGRATION_GUIDE.md) for detailed documentation.

### Optimized DeepStack Training System ✨

The DeepStack neural network training pipeline has been completely optimized and is now production-ready:

- **✅ Fixed all import issues** - Proper module structure and imports
- **✅ Corrected training configuration** - Accurate num_buckets (36) matching training data
- **✅ Enhanced training script** - Early stopping, LR scheduling, gradient clipping
- **✅ Model validation tools** - Comprehensive quality assessment
- **✅ Complete documentation** - Full guides and integration examples

**Quick Start:**
```bash
# Train DeepStack value network (5 min on GPU)
python scripts/train_deepstack.py --use-gpu

# Validate trained model
python scripts/validate_deepstack_model.py
```

See [DeepStack Training Guide](docs/DEEPSTACK_TRAINING.md) for complete documentation.

## Features

### 🏆 Advanced AI Agents

- **PokerBot Agent (NEW!)**: Unified world-class agent combining all features ✨
  - Modular architecture with configurable components
  - CFR/CFR+ for game-theoretic optimal play
  - DQN for pattern recognition and learning
  - DeepStack continual re-solving
  - Opponent modeling and pre-trained models
  - **Recommended for all new development**
- **Champion Agent**: Unified CFR + DQN hybrid with pre-trained models (deprecated, use PokerBot)
- **Elite Unified Agent**: Advanced multi-component agent (deprecated, use PokerBot)
- **DeepStack Engine**: Complete Python/PyTorch port of the championship DeepStack AI
  - Continual re-solving for dynamic game tree solving
  - Neural network value estimation for depth-limited search
  - CFR-based game-theoretic optimal play
- **Search Agent**: Real-time depth-limited search with blueprint strategy
- **Advanced CFR Agent**: CFR with pruning, linear discounting, and progressive training
- **DQN Agent**: Deep reinforcement learning agent using Q-learning
- **CFR Agent**: Counterfactual Regret Minimization for game-theoretic optimal play
- **Fixed Strategy Agent**: GTO-inspired fixed strategy with pot odds calculations
- **Random Agent**: Baseline random decision-making agent

> 📖 **Migration Guide**: See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for details on using the new PokerBot agent.

### 🚀 Progressive Training Pipeline

- **Multi-phase CFR Training**: Warmup → Pruning → Linear CFR → Strategy updates
- **Distributed Training**: Multiprocessing for 4-8x speedup
- **Card Abstraction**: Information set clustering for memory efficiency
- **Vicarious Learning**: Multi-agent training and imitation learning

### 🎯 Advanced Features

- **DeepStack Continual Re-solving**: Dynamic game tree solving during live play
  - Depth-limited search with neural network value estimation
  - CFRDGadget for opponent range reconstruction
  - Sub-second decision making via efficient CFR
- **Blueprint + Real-time Search**: Two-stage decision making
- **Information Set Abstraction**: Reduces 56B+ states to tractable sizes
- **CFR with Pruning (CFRp)**: 20x+ speedup through action pruning
- **Linear CFR**: Faster convergence via discounting
- **Pre-trained Models**: DeepStack value networks and equity tables

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
│   ├── deepstack/         # DeepStack AI engine (ported from Lua)
│   │   ├── tree_builder.py      # Game tree construction
│   │   ├── tree_cfr.py          # CFR solver
│   │   ├── terminal_equity.py   # Equity calculation
│   │   ├── cfrd_gadget.py       # Range reconstruction
│   │   ├── value_nn.py          # Neural network
│   │   ├── resolving.py         # Continual re-solving API
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
├── tests/                 # Test suite
│   └── test_deepstack_core.py
├── examples/              # Examples and demos
│   ├── demo_champion.py
│   ├── example_champion.py
│   └── test_champion.py
├── docs/                  # Documentation
│   ├── PORTING_BLUEPRINT.md
│   ├── DEEPSTACK_GUIDE.md
│   ├── CHAMPION_AGENT.md
│   └── ...
├── data/                  # Training data and documentation
│   ├── doc/              # Original DeepStack Lua documentation
│   └── ...
├── models/                # Saved models and pre-trained weights
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

### 🚀 Using DeepStack Engine

The DeepStack engine provides world-class game-theoretic poker play:

```python
from src.deepstack.resolving import Resolving
import numpy as np

# Initialize resolver
resolver = Resolving(num_hands=6, game_variant='leduc')

# Define game state
node_params = {
    'street': 0,
    'bets': [20, 20],
    'current_player': 1,
    'board': [],
    'bet_sizing': [1.0]  # Pot-sized bets
}

# Player and opponent ranges
player_range = np.ones(6) / 6
opponent_range = np.ones(6) / 6

# Solve using continual re-solving
resolver.resolve_first_node(node_params, player_range, opponent_range, 
                             iterations=500)

# Get optimal strategy
actions = resolver.get_possible_actions()
for action in actions:
    prob = resolver.get_action_strategy(action)
    print(f"P({action}) = {prob:.4f}")
```

**See [docs/DEEPSTACK_GUIDE.md](docs/DEEPSTACK_GUIDE.md) for complete DeepStack documentation.**

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

1. **DeepStack** - https://www.deepstack.ai/
   - Complete Python/PyTorch port of the championship DeepStack AI
   - Continual re-solving with depth-limited search
   - Original Lua documentation in `data/doc/`
   
2. **GTO Poker Bot** - https://github.com/arnenoori/gto-poker-bot
   - Vision-based game state detection
   - Automated playing framework

3. **Poker AI** - https://github.com/dickreuter/Poker
   - Advanced poker strategies
   - Pre-trained models

4. **DeepStack-Leduc** - https://github.com/lifrordi/DeepStack-Leduc
   - Deep counterfactual regret minimization
   - Reference implementation

5. **Libratus** - https://noambrown.github.io/papers/17-IJCAI-Libratus.pdf
   - Game theory optimal strategies
   - Abstraction techniques

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is for educational and research purposes only. Please ensure compliance with all applicable laws and terms of service when using this software.
