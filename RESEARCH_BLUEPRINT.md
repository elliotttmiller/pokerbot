# RESEARCH BLUEPRINT: POKER SMOKER PROJECT
## Project Genesis - Architecture & Design Document

**Document Version:** 1.0  
**Date:** 2025-10-14  
**Architect:** Autonomous Coding Agent  
**Classification:** Genesis Mandate - First Principles Research

---

## EXECUTIVE SUMMARY

The Poker Smoker project represents a synthesis of cutting-edge AI poker research into a single, unified, state-of-the-art poker bot system. This blueprint documents the architectural decisions, research findings, and implementation strategy for building an advanced No-Limit Texas Hold'em AI agent that combines Game Theory Optimal (GTO) play with exploitative adaptation.

**Core Innovation:** A hybrid architecture that fuses deep neural network value estimation with real-time counterfactual regret minimization (CFR) solving, augmented with opponent modeling for exploitative play.

---

## PART I: RESEARCH CORPUS ANALYSIS

### 1. DeepStack Paper - The Theoretical Foundation

**Key Insights:**
- **Continual Re-solving:** Rather than computing a full game strategy upfront, DeepStack solves subgames on-the-fly during play
- **Deep Counterfactual Value Networks:** Neural networks approximate the values of different actions at decision points, replacing expensive CFR traversals
- **Limited Lookahead Search:** The agent plans only a few moves ahead, using value networks to estimate future payoffs
- **Sound Decomposition:** Mathematically proven to be sound - guaranteed not to lose against perfect opponents

**Architectural Takeaways:**
1. Use neural networks for value estimation, not full policy networks
2. Implement a real-time solver that uses these value estimates
3. Focus on subgame solving rather than precomputed strategies
4. Maintain soundness guarantees through proper CFR implementation

### 2. DeepStack-Leduc - The GTO Engine

**Key Insights:**
- **CFR+ Algorithm:** An improved variant of CFR with faster convergence
- **Public State Trees:** Efficient representation of poker game states using public information
- **Range Representation:** Players are represented as probability distributions over possible hands (ranges)
- **Terminal Node Evaluation:** Uses neural networks to estimate expected values at terminal nodes of lookahead trees

**Architectural Takeaways:**
1. Implement CFR+ as the core solving algorithm
2. Use public belief states for efficient game tree representation
3. Represent player strategies as hand ranges (probability distributions)
4. Build modular components: game engine, CFR solver, value network

### 3. g5-poker-bot - The Exploitative Mind

**Key Insights:**
- **Opponent Modeling:** Tracks opponent betting patterns and adjusts strategy
- **Exploitative Play:** Identifies and exploits opponent weaknesses (too loose, too tight, too aggressive)
- **Hand Strength Assessment:** Uses Monte Carlo simulation to estimate equity
- **Adaptive Betting:** Adjusts bet sizing based on opponent tendencies

**Architectural Takeaways:**
1. Implement opponent modeling module to track betting statistics
2. Build exploitability detector to identify weaknesses
3. Create strategy adjustment layer that shifts from GTO toward exploitation
4. Use Monte Carlo equity calculation for hand strength estimation

### 4. self-operating-computer - The Autonomous Body

**Key Insights:**
- **Vision-Language Model Integration:** Uses multimodal AI to understand GUI state
- **Action Execution Framework:** Translates decisions into concrete actions (clicks, types)
- **World Model:** Maintains internal representation of the application state
- **Robust Error Handling:** Gracefully handles unexpected states

**Architectural Takeaways:**
1. Design for both headless (training) and headed (live play) operation
2. Create abstract action interface that can be implemented for GUI or simulation
3. Build state perception module for reading game state
4. Implement robust error recovery and logging

### 5. gto-poker-bot - The Modern Implementation

**Key Insights:**
- **Modern Python Stack:** Uses PyTorch, NumPy, and modern libraries
- **ONNX Model Export:** Enables cross-platform model deployment
- **Modular Architecture:** Clean separation of concerns (game, solver, agent, training)
- **Configuration-Driven:** Uses config files for hyperparameters and settings

**Architectural Takeaways:**
1. Use Python 3.11+ with modern type hints and async/await
2. Export models to ONNX format for production deployment
3. Implement clean modular architecture with clear interfaces
4. Use configuration files for all tunable parameters

---

## PART II: MASTER ARCHITECTURAL DECISIONS

### THE MIND - Cognitive Architecture

**Decision:** Hybrid GTO-Exploitative Architecture with Real-Time Solving

The system consists of three cognitive layers:

1. **GTO Foundation Layer**
   - Deep value network trained via self-play CFR
   - Provides baseline unexploitable strategy
   - Guarantees minimum performance floor

2. **Real-Time Solver Layer**
   - CFR+ solver for subgame resolution
   - Uses value network for terminal node evaluation
   - Computes optimal strategy for current situation

3. **Exploitative Adaptation Layer**
   - Opponent modeling engine tracks player tendencies
   - Adjusts strategy to exploit identified weaknesses
   - Blends with GTO strategy based on confidence level

**State Representation:**
```
Observation Space = {
    hand_cards: [2D vector (rank, suit) × 2 cards],
    board_cards: [2D vector × 0-5 cards],
    pot_size: [normalized scalar],
    stack_sizes: [normalized vector × num_players],
    current_bets: [normalized vector × num_players],
    position: [one-hot encoded],
    betting_history: [sequence of actions],
    opponent_stats: [aggregate statistics vector]
}
```

**Rationale:** This representation captures all decision-relevant information while remaining tractable for neural network processing.

### THE FORGE - Training Pipeline

**Decision:** Self-Play CFR with Deep Supervised Learning

**Training Architecture:**
1. **Data Generation Phase**
   - Run CFR+ self-play iterations on poker game tree
   - Generate training examples (state, value) pairs
   - Store in replay buffer with prioritized sampling

2. **Network Training Phase**
   - Train deep value network on generated data
   - Use supervised learning with regression loss
   - Implement curriculum learning (simple → complex scenarios)

3. **Evaluation & Iteration Phase**
   - Evaluate new network vs previous generation
   - Test against fixed opponents (GTO baseline, exploitable bots)
   - Checkpoint best-performing models

**Virtuous Cycle:**
```
Generate Data (CFR) → Train Network → Evaluate → 
Use Improved Network for Next Generation → Repeat
```

**Training Regimen:**
- **Network Architecture:** Multi-layer Transformer or deep MLP
  - Input: Embedded game state (~256 dimensions)
  - Hidden: 3-5 layers of 512-1024 units with residual connections
  - Output: Value estimation for each action
- **Optimization:** Adam optimizer with learning rate scheduling
- **Batch Size:** 256-512 examples
- **Training Data:** 10M+ game states per generation

### THE BODY - Autonomous Operation

**Decision:** Dual-Mode Agent with Abstract Action Interface

**Architecture:**
```
Agent Core (decision logic)
    ↓
Action Interface (abstract)
    ↓
    ├─→ Simulation Mode (for training)
    └─→ Live Mode (for GUI interaction)
```

**Key Components:**
1. **Perception Module**
   - Headless: Receives game state directly from simulator
   - Headed: Parses GUI state (screen capture + OCR if needed)

2. **Decision Module**
   - Loads ONNX model
   - Runs inference to compute optimal action
   - Applies opponent model adjustments

3. **Execution Module**
   - Headless: Returns action to simulator
   - Headed: Executes GUI actions (click, type)

4. **Memory Module**
   - Maintains game history
   - Updates opponent models
   - Logs decisions for analysis

---

## PART III: IMPLEMENTATION STRATEGY

### Technology Stack

**Core Technologies:**
- **Language:** Python 3.11+
- **Deep Learning:** PyTorch 2.0+
- **Numerical Computing:** NumPy, SciPy
- **Model Deployment:** ONNX Runtime
- **Environment:** Gymnasium (OpenAI Gym successor)
- **Configuration:** YAML + dataclasses
- **Logging:** Structured logging with loguru
- **Testing:** pytest

**Optional (for GUI interaction):**
- **Computer Vision:** OpenCV, pytesseract (OCR)
- **GUI Automation:** pyautogui, python-xlib

### Directory Structure

```
pokerbot/
├── README.md                    # Comprehensive project documentation
├── RESEARCH_BLUEPRINT.md        # This document
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── config/
│   ├── training_config.yaml    # Training hyperparameters
│   ├── agent_config.yaml       # Agent runtime config
│   └── game_config.yaml        # Game rules configuration
├── src/
│   ├── __init__.py
│   ├── engine/                 # Poker game engine
│   │   ├── __init__.py
│   │   ├── game_state.py      # Game state representation
│   │   ├── rules.py           # NLHE rules implementation
│   │   ├── equity.py          # Hand equity calculation
│   │   └── hand_evaluator.py # Hand strength evaluation
│   ├── solver/                # CFR solver components
│   │   ├── __init__.py
│   │   ├── cfr.py            # CFR+ algorithm
│   │   ├── nodes.py          # Game tree nodes
│   │   └── ranges.py         # Range representation
│   ├── models/               # Neural network models
│   │   ├── __init__.py
│   │   ├── value_network.py  # Deep value network
│   │   ├── architectures.py  # Network architectures
│   │   └── export.py         # ONNX export utilities
│   ├── training/             # Training pipeline
│   │   ├── __init__.py
│   │   ├── data_generation.py # CFR data generation
│   │   ├── trainer.py        # Network training loop
│   │   ├── replay_buffer.py  # Experience replay
│   │   └── evaluator.py      # Model evaluation
│   ├── agent/                # Autonomous agent
│   │   ├── __init__.py
│   │   ├── decision_maker.py # Core decision logic
│   │   ├── opponent_model.py # Opponent modeling
│   │   ├── perception.py     # State perception
│   │   └── executor.py       # Action execution
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── config.py         # Configuration loading
│       ├── logger.py         # Logging setup
│       └── metrics.py        # Performance metrics
├── scripts/
│   ├── train.py              # Training pipeline script
│   ├── evaluate.py           # Model evaluation script
│   └── run_agent.py          # Live agent runner
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_engine.py
│   ├── test_solver.py
│   ├── test_models.py
│   └── test_agent.py
├── models/                   # Saved model artifacts
│   └── .gitkeep
└── data/                     # Training data
    └── .gitkeep
```

### Implementation Phases

**Phase 1: Foundation (Core Engine)**
- Implement poker game state representation
- Build NLHE rules engine
- Create hand evaluator and equity calculator
- Write comprehensive unit tests

**Phase 2: Solver (CFR Implementation)**
- Implement CFR+ algorithm
- Build game tree nodes and public belief states
- Create range representation and operations
- Test solver convergence on toy games

**Phase 3: Neural Networks (Value Network)**
- Design and implement value network architecture
- Create training data format
- Implement ONNX export functionality
- Validate network can learn simple poker scenarios

**Phase 4: Training Pipeline**
- Build CFR data generation system
- Implement training loop with checkpointing
- Create evaluation framework
- Generate first "Genesis" model

**Phase 5: Agent (Decision Making)**
- Implement core decision logic using value network
- Build opponent modeling system
- Create exploitative adjustment layer
- Test agent in simulation

**Phase 6: Integration & Deployment**
- Create run_agent.py for live play
- Build perception and execution modules
- Implement robust error handling
- Comprehensive testing and documentation

---

## PART IV: KEY INNOVATIONS & DIFFERENTIATORS

### 1. Hybrid GTO-Exploitative Architecture
**Innovation:** Seamless blending between unexploitable baseline and exploitative adaptation based on opponent model confidence.

**Implementation:**
```python
action = (1 - exploit_weight) * gto_action + exploit_weight * exploit_action
where exploit_weight = f(opponent_model_confidence, risk_tolerance)
```

### 2. Efficient State Representation
**Innovation:** Transformer-compatible state encoding that captures sequential betting history while remaining computationally tractable.

**Benefit:** Enables the network to learn complex betting patterns and meta-game dynamics.

### 3. Curriculum-Based Training
**Innovation:** Progressive training from simplified game variants to full NLHE complexity.

**Curriculum:**
1. Headsup, single street (river only)
2. Headsup, two streets (turn + river)
3. Headsup, full game (preflop → river)
4. 3-player, full game
5. 6-player (final), full game

### 4. Real-Time Subgame Solving
**Innovation:** On-demand CFR solving of current subgame using value network for lookahead terminal evaluation.

**Benefit:** Combines computational efficiency with strategic sophistication - doesn't require precomputed strategies.

### 5. Model Generations with Competitive Selection
**Innovation:** Each training cycle produces a new generation that must defeat all previous generations to become the new champion.

**Benefit:** Ensures monotonic improvement and prevents catastrophic forgetting.

---

## PART V: PERFORMANCE TARGETS & SUCCESS METRICS

### Training Targets
- **Convergence:** CFR exploitability < 1% of pot after 10K iterations
- **Network Accuracy:** Value prediction MSE < 0.05 on holdout set
- **Training Time:** Genesis model achievable in < 24 hours on single GPU
- **Model Size:** Final ONNX model < 50MB for production deployment

### Agent Performance Targets
- **vs GTO Baseline:** Win rate ≥ -5bb/100 (near-break-even)
- **vs Exploitable Opponents:** Win rate ≥ +15bb/100
- **Decision Time:** < 5 seconds per decision on consumer hardware
- **Robustness:** 99%+ successful action execution rate

### Code Quality Targets
- **Test Coverage:** ≥ 80% line coverage
- **Type Safety:** Full type hints with mypy strict mode
- **Documentation:** Comprehensive docstrings and README
- **Modularity:** Clean separation of concerns, dependency injection

---

## PART VI: RISK MITIGATION & CONTINGENCIES

### Technical Risks

**Risk 1: Neural Network Training Instability**
- **Mitigation:** Implement gradient clipping, learning rate scheduling, early stopping
- **Contingency:** Fall back to simpler network architecture (fewer layers/units)

**Risk 2: CFR Convergence Issues**
- **Mitigation:** Use proven CFR+ variant, test on simple games first
- **Contingency:** Implement Monte Carlo CFR for better sampling efficiency

**Risk 3: Computational Resource Constraints**
- **Mitigation:** Optimize code with JIT compilation, use efficient data structures
- **Contingency:** Reduce game complexity (fewer players, limit bet sizes)

**Risk 4: Opponent Modeling Overfitting**
- **Mitigation:** Require minimum sample size, blend with GTO cautiously
- **Contingency:** Disable exploitative layer, operate in pure GTO mode

### Implementation Risks

**Risk 1: Scope Creep**
- **Mitigation:** Strict adherence to MVP feature set, defer enhancements
- **Contingency:** Implement phased delivery, core functionality first

**Risk 2: Integration Complexity**
- **Mitigation:** Define clear interfaces between modules, extensive unit testing
- **Contingency:** Simplify architecture, reduce number of components

---

## PART VII: FUTURE ENHANCEMENTS (POST-MVP)

### Near-Term Enhancements
1. **Multi-Table Tournament (MTT) Support:** Adapt strategy for ICM considerations
2. **Advanced Opponent Models:** Clustering similar players, long-term memory
3. **GUI Integration:** Direct integration with popular poker clients
4. **Bankroll Management:** Risk-of-ruin analysis and stake selection

### Long-Term Research Directions
1. **Multi-Agent RL:** Training against diverse opponent populations
2. **Transfer Learning:** Pre-trained models on related poker variants
3. **Explainable AI:** Visualizing decision rationale for human learning
4. **Human-AI Collaboration:** Providing real-time advice to human players

---

## CONCLUSION

The Poker Smoker project synthesizes the best ideas from cutting-edge poker AI research into a cohesive, state-of-the-art system. By combining:
- DeepStack's sound subgame solving approach
- Efficient CFR algorithms from DeepStack-Leduc
- Exploitative adaptation from g5-poker-bot
- Modern engineering practices from gto-poker-bot
- Autonomous operation concepts from self-operating-computer

We create a poker bot that is theoretically sound, practically effective, and architecturally elegant.

The system is designed from first principles to be:
- **Sound:** Guaranteed not to be exploited by perfect opponents
- **Strong:** Capable of exploiting weak opponents effectively
- **Scalable:** Trainable on consumer hardware, deployable efficiently
- **Maintainable:** Clean, modular, well-documented codebase
- **Extensible:** Easy to enhance with new features and techniques

This blueprint serves as the constitutional document for the Poker Smoker project, defining not just what to build, but why each decision was made and how it serves the ultimate goal: creating the most advanced poker AI system possible.

**Mission Status:** Genesis Mandate Initiated. The foundation is laid. Now we build.

---

*Document Classification: GENESIS MANDATE - FIRST PRINCIPLES RESEARCH*  
*Next Step: Implementation Phase 1 - Core Engine Development*
