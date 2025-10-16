# DeepStack Porting Blueprint

## Executive Summary

This document details the comprehensive plan to port the official DeepStack poker AI engine from Lua/Torch7 to Python/PyTorch and integrate it into our existing pokerbot framework.

**Source**: DeepStack Lua documentation in `data/doc/`  
**Target**: Complete Python/PyTorch implementation in `src/deepstack/`  
**Status**: Stub implementations exist; full porting required

---

## 1. Architecture Analysis

### 1.1 Original DeepStack Architecture (Lua/Torch7)

DeepStack is structured into several key subsystems:

#### Core Modules (Source/)
- **Game/** - Poker game mechanics (cards, board, hand evaluation)
- **TerminalEquity/** - Terminal state evaluation
- **Tree/** - Public game tree construction
- **Lookahead/** - Depth-limited CFR solver (primary game solver)
- **Nn/** - Neural network interface and training
- **DataGeneration/** - Training data generation via self-play
- **Training/** - Neural network training loop
- **Player/** - ACPC player with continual re-solving
- **ACPC/** - ACPC protocol communication

#### Key Algorithms
1. **Continual Re-solving** - Dynamic game tree solving during play
2. **Depth-Limited Search** - Limit search depth and use neural net for leaf values
3. **CFR (Counterfactual Regret Minimization)** - Game-theoretic solver
4. **Neural Net Value Estimation** - Replace leaf evaluation with learned values

### 1.2 Current Python Implementation Status

**Existing in src/deepstack/:**
- ✅ `arguments.py` - Config (stub, 12 lines)
- ⚠️ `bucketer.py` - Card bucketing (stub, 14 lines)
- ✅ `card_abstraction.py` - Card clustering (289 lines, functional)
- ⚠️ `cfrd_gadget.py` - CFR decomposition (stub, 16 lines)
- ⚠️ `data_stream.py` - Data loading (stub, 40 lines)
- ⚠️ `deepstack_trainer.py` - Training (stub, 68 lines)
- ✅ `evaluator.py` - Hand evaluation (171 lines, functional)
- ✅ `hand_evaluator.py` - Hand strength (205 lines, functional)
- ⚠️ `masked_huber_loss.py` - Loss function (stub, 13 lines)
- ✅ `monte_carlo.py` - MC simulation (276 lines, functional)
- ⚠️ `net_builder.py` - Network builder (stub, 29 lines)
- ⚠️ `strategy_filling.py` - Strategy filling (stub, 25 lines)
- ⚠️ `terminal_equity.py` - Terminal equity (stub, 34 lines)
- ⚠️ `train_deepstack.py` - Training script (stub, 30 lines)
- ⚠️ `tree_builder.py` - Tree builder (stub, 64 lines)
- ⚠️ `tree_cfr.py` - Tree CFR (stub, 19 lines)

**Legend**: ✅ Functional | ⚠️ Stub/Incomplete

---

## 2. Core Components to Port

### 2.1 Priority 1: Lookahead (The Heart of DeepStack)

**Source**: `data/doc/classes/lookahead.html`, `data/doc/manual/internals.md`  
**Target**: `src/deepstack/lookahead.py`

The Lookahead class is the core CFR solver that uses tensors for efficient computation.

#### Key Concepts from Documentation:
- Uses multi-dimensional tensors to represent game tree layers
- First 3 dimensions: `[action_id, parent_action_id, grandparent_id]`
- Enables efficient CFR up/down traversal via tensor operations
- Stores per-layer data: `players` and `cards` dimensions
- Masks padding nodes that don't correspond to real game states

#### Methods to Implement:
```python
class Lookahead:
    def __init__(self):
        """Constructor"""
        
    def build_lookahead(self, tree):
        """Constructs the lookahead from a game's public tree.
        
        Converts tree structure into efficient tensor representation.
        Must be called before resolve methods.
        
        Args:
            tree: Public tree built by PokerTreeBuilder
        """
        
    def resolve_first_node(self, player_range, opponent_range):
        """Re-solves the lookahead using input ranges.
        
        Used for root node where ranges are fixed (uniform or specified).
        Runs CFR iterations until convergence.
        
        Args:
            player_range: Range vector for re-solving player
            opponent_range: Range vector for opponent
        """
        
    def resolve(self, player_range, opponent_cfvs):
        """Re-solves using player range and CFRDGadget for opponent.
        
        Used during gameplay after opponent actions.
        Opponent range generated via CFRDGadget from their CFVs.
        
        Args:
            player_range: Range vector for re-solving player
            opponent_cfvs: Counterfactual values for opponent
        """
        
    def get_chance_action_cfv(self, action_index, board):
        """Gets opponent CFVs after chance event (board cards dealt).
        
        Args:
            action_index: Index of chance action
            board: Board cards
            
        Returns:
            Opponent counterfactual values
        """
        
    def get_results(self):
        """Gets results of re-solving.
        
        Returns:
            dict with 'strategy', 'root_cfvs', 'children_cfvs'
        """
```

#### Tensor Representation Details:
- Separate tensor per tree layer (betting round)
- Dimensions: `[actions, parent_actions, grandparents, players, cards]`
- Example operations:
  - Select all fold nodes: `nodes[0]` (fold is first action)
  - Select all call nodes: `nodes[1]` (call is second action)
  - Select call after all-in: `nodes[1][-1]` (call after last parent action)

### 2.2 Priority 2: Resolving (Continual Re-solving API)

**Source**: `data/doc/classes/resolving.html`  
**Target**: `src/deepstack/resolving.py`

High-level API for continual re-solving during gameplay. Wraps Lookahead class.

#### Methods to Implement:
```python
class Resolving:
    def __init__(self):
        """Constructor"""
        
    def resolve_first_node(self, node, player_range, opponent_range):
        """Re-solve depth-limited lookahead at root node.
        
        Builds lookahead tree from node and runs CFR.
        
        Args:
            node: Current game node
            player_range: Player's range
            opponent_range: Opponent's range
        """
        
    def resolve(self, node, player_range, opponent_cfvs):
        """Re-solve using CFRDGadget for opponent range.
        
        Args:
            node: Current game node
            player_range: Player's range
            opponent_cfvs: Opponent's CFVs from previous solve
        """
        
    def get_possible_actions(self):
        """List of actions at current node"""
        
    def get_root_cfv(self):
        """Average CFVs for re-solve player at root"""
        
    def get_root_cfv_both_players(self):
        """Average CFVs for both players at root"""
        
    def get_action_cfv(self, action):
        """Opponent CFVs after player takes action"""
        
    def get_chance_action_cfv(self, action, board):
        """Opponent CFVs after chance event"""
        
    def get_action_strategy(self, action):
        """Probability of taking action in strategy"""
```

### 2.3 Priority 3: TreeCFR (Full Tree Solver)

**Source**: `data/doc/classes/tree_cfr.html`, `data/doc/manual/tutorial.md`  
**Target**: `src/deepstack/tree_cfr.py` (enhance existing stub)

Full CFR solver for small trees. Used for validation and small game solving.

#### Methods to Implement:
```python
class TreeCFR:
    def __init__(self):
        """Constructor"""
        
    def run_cfr(self, tree, starting_ranges, iterations=1000):
        """Runs CFR on tree to compute nash equilibrium strategy.
        
        After 1000 iterations on tutorial tree, exploitability should be ~1.0 chips.
        
        Args:
            tree: Public tree from PokerTreeBuilder
            starting_ranges: Starting ranges for both players
            iterations: Number of CFR iterations
        """
```

#### CFR Algorithm:
1. Initialize regrets and strategies to zero
2. For each iteration:
   - Traverse tree from root to leaves
   - Compute counterfactual values at each node
   - Update regrets based on alternative actions
   - Update strategy using regret matching
3. Average strategies across iterations
4. Return final strategy

### 2.4 Priority 4: ValueNN (Neural Network Interface)

**Source**: `data/doc/classes/value_nn.html`, `data/doc/modules/net_builder.html`  
**Target**: `src/deepstack/value_nn.py`

Neural network wrapper for value estimation at lookahead leaves.

#### Network Architecture (from tutorial):
```lua
-- Original Lua config
params.net = '{nn.Linear(input_size, 50), nn.PReLU(), nn.Linear(50, output_size)}'
-- 5 hidden layers, 50 neurons each for production model
```

#### Python/PyTorch Implementation:
```python
class ValueNN:
    def __init__(self, model_path=None):
        """Constructor. Loads neural net from disk.
        
        Args:
            model_path: Path to saved model (default: 'models/pretrained/final.pt')
        """
        
    def get_value(self, inputs):
        """Neural net forward pass.
        
        Args:
            inputs: NxI tensor, N instances of inputs
            
        Returns:
            NxO tensor, N sets of outputs
        """
```

#### Input/Output Format (from net_builder):
- **Input**: `[player_range, opponent_range, pot_size]`
  - player_range: Vector of probabilities for each hand bucket
  - opponent_range: Vector of probabilities for each hand bucket  
  - pot_size: Normalized pot size
  - Total input size: 2 * num_buckets + 1
  
- **Output**: `[player_values, opponent_values]`
  - player_values: Expected values for each player hand bucket
  - opponent_values: Expected values for each opponent hand bucket
  - Total output size: 2 * num_buckets
  - Uses **masked Huber loss** for training

### 2.5 Priority 5: Supporting Modules

#### TreeBuilder (Enhanced)
**Target**: `src/deepstack/tree_builder.py` (enhance existing)

- Build complete game tree from root node
- Support bet sizing abstractions
- Handle terminal nodes (fold, showdown)
- Track pot, bets, current player
- Generate children recursively

#### TreeStrategyFilling  
**Target**: `src/deepstack/tree_strategy_filling.py`

```python
class TreeStrategyFilling:
    def fill_strategies(self, tree, player, player_range, opponent_range):
        """Fills tree nodes with strategies via continual re-solving.
        
        Traverses tree and re-solves at each node.
        Tutorial shows exploitability of ~1.37 chips for example tree.
        
        Args:
            tree: Public tree
            player: Player index (1 or 2)
            player_range: Player's range
            opponent_range: Opponent's range
        """
```

#### TerminalEquity (Enhanced)
**Target**: `src/deepstack/terminal_equity.py` (enhance existing)

```python
class TerminalEquity:
    def set_board(self, board):
        """Set board cards for equity calculations"""
        
    def get_call_matrix(self):
        """Returns equity matrix for call nodes.
        
        Matrix[i][j] = equity of player hand i vs opponent hand j
        """
        
    def get_fold_matrix(self):
        """Returns payoff matrix for fold nodes"""
```

#### CFRDGadget (Enhanced)
**Target**: `src/deepstack/cfrd_gadget.py` (enhance existing)

Counterfactual value decomposition for opponent range reconstruction.

```python
class CFRDGadget:
    def __init__(self, board, player_range, opponent_cfvs):
        """Initialize gadget for opponent range computation.
        
        Args:
            board: Board cards
            player_range: Player's range
            opponent_cfvs: Opponent's counterfactual values
        """
        
    def compute_opponent_range(self):
        """Computes opponent range from CFVs.
        
        Returns:
            Opponent range vector
        """
```

#### MaskedHuberLoss (Enhanced)
**Target**: `src/deepstack/masked_huber_loss.py` (enhance existing)

```python
def masked_huber_loss(y_true, y_pred, mask, delta=1.0):
    """Huber loss with masking for invalid hand combinations.
    
    Used in neural net training to ignore impossible hand pairs.
    
    Args:
        y_true: Target values
        y_pred: Predicted values  
        mask: Binary mask (1 = valid, 0 = invalid)
        delta: Huber loss threshold
        
    Returns:
        Scalar loss value
    """
```

---

## 3. Integration Strategy

### 3.1 Champion Agent Integration

**Current State**: `src/agents/champion_agent.py` has placeholder integration

**Changes Required**:

1. **Replace stub DeepStackLookahead** with full Lookahead + Resolving:
```python
from src.deepstack.lookahead import Lookahead
from src.deepstack.resolving import Resolving
from src.deepstack.value_nn import ValueNN

class ChampionAgent(BaseAgent):
    def __init__(self, ...):
        # Replace stub lookahead
        self.resolving = Resolving()
        self.value_nn = ValueNN(model_path='models/deepstack/final.pt')
```

2. **Update decide_action** to use continual re-solving:
```python
def decide_action(self, game_state):
    # Build current node from game_state
    node = self._build_node(game_state)
    
    # Get ranges
    my_range = self._get_my_range(game_state)
    opp_cfvs = self._get_opponent_cfvs(game_state)
    
    # Re-solve
    if game_state.street == 0 and len(game_state.history) == 0:
        # Root node - use uniform ranges
        opp_range = self._get_uniform_range()
        self.resolving.resolve_first_node(node, my_range, opp_range)
    else:
        # Use CFRDGadget for opponent range
        self.resolving.resolve(node, my_range, opp_cfvs)
    
    # Get strategy
    actions = self.resolving.get_possible_actions()
    strategies = [self.resolving.get_action_strategy(a) for a in actions]
    
    # Sample action from strategy
    action = np.random.choice(actions, p=strategies)
    return action
```

3. **Integrate with existing CFR/DQN**:
- Keep CFR for preflop strategy
- Keep DQN for opponent modeling
- Use DeepStack for in-depth solving

### 3.2 Training Pipeline Integration

**Target**: `scripts/train_deepstack.py` (new) and `src/deepstack/deepstack_trainer.py`

1. **Data Generation**:
```python
from src.deepstack.data_generation import DataGenerator

generator = DataGenerator()
generator.generate_training_data(
    num_samples=100000,
    output_dir='data/train_samples'
)
```

2. **Model Training**:
```python
from src.deepstack.trainer import DeepStackTrainer

trainer = DeepStackTrainer(
    model_path='models/deepstack',
    data_path='data/train_samples'
)
trainer.train(epochs=50, batch_size=128)
```

3. **Model Evaluation**:
```python
from src.deepstack.tree_strategy_filling import TreeStrategyFilling

# Build test tree
tree = builder.build_tree(params)

# Fill with strategies
filling = TreeStrategyFilling()
filling.fill_strategies(tree, 1, range1, range2)

# Compute exploitability
exploitability = compute_exploitability(tree)
print(f"Exploitability: {exploitability} chips")
```

---

## 4. Technical Specifications

### 4.1 Lua to Python/PyTorch Mapping

| Lua/Torch7 | Python/PyTorch | Notes |
|------------|----------------|-------|
| `torch.Tensor` | `torch.Tensor` | Similar API |
| `tensor:copy(x)` | `tensor.copy_(x)` | In-place copy |
| `tensor[i]` | `tensor[i-1]` | Lua is 1-indexed |
| `tensor:size(dim)` | `tensor.size(dim)` | Same |
| `tensor:zero()` | `tensor.zero_()` | In-place zeroing |
| `torch.dot(a, b)` | `torch.dot(a, b)` | Same |
| `tensor:sum(dim)` | `tensor.sum(dim=dim)` | Named arg |
| `nn.Linear(in, out)` | `nn.Linear(in, out)` | Same |
| `nn.PReLU()` | `nn.PReLU()` | Same |
| `cutorch` (GPU) | `tensor.cuda()` | Similar |

### 4.2 Game Representation

**Cards**:
- Leduc: 6 cards (Jack, Queen, King × 2 suits)
- Texas Hold'em: 52 cards (13 ranks × 4 suits)
- Represented as integers 0-5 (Leduc) or 0-51 (Hold'em)

**Ranges**:
- Vector of probabilities for each possible hand
- Size = number of possible private hands
- Sums to 1.0 for valid range

**Actions**:
- Fold: -2
- Call/Check: -1  
- Raise: 1+ (multiple bet sizes possible)

**Node Types**:
- Terminal fold: -2
- Terminal call: -1
- Chance node: 0
- Inner node: 2

### 4.3 Neural Network Specifications

**Architecture** (Leduc):
```
Input: 2 * 6 + 1 = 13 features
  - Player range: 6 hand probabilities
  - Opponent range: 6 hand probabilities
  - Pot size: 1 normalized value

Hidden: [50, 50, 50, 50, 50] neurons (5 layers)
Activation: PReLU

Output: 2 * 6 = 12 values
  - Player hand values: 6 values
  - Opponent hand values: 6 values
```

**Architecture** (Texas Hold'em):
```
Input: 2 * 169 + 1 = 339 features
  - Player range: 169 hand bucket probabilities
  - Opponent range: 169 hand bucket probabilities
  - Pot size: 1 normalized value

Hidden: [512, 512, 512, 512] neurons (4 layers)
Activation: PReLU

Output: 2 * 169 = 338 values
  - Player hand bucket values: 169 values
  - Opponent hand bucket values: 169 values
```

**Loss**: Masked Huber Loss
- Delta: 1.0
- Mask: 1 for valid hand combinations, 0 for impossible pairs

**Optimizer**: Adam
- Learning rate: 0.001
- Batch size: 128

---

## 5. Testing Strategy

### 5.1 Unit Tests

Create tests for each ported module:

```python
# tests/deepstack/test_lookahead.py
def test_lookahead_build():
    """Test lookahead construction from tree"""
    
def test_lookahead_resolve_first_node():
    """Test root node re-solving"""
    
def test_lookahead_resolve():
    """Test re-solving with CFRDGadget"""

# tests/deepstack/test_tree_cfr.py  
def test_tree_cfr_convergence():
    """Test CFR converges to low exploitability"""
    # After 1000 iterations, exploitability should be ~1.0 chips
    
# tests/deepstack/test_value_nn.py
def test_value_nn_forward():
    """Test neural net forward pass"""
    
def test_value_nn_load():
    """Test loading saved model"""
```

### 5.2 Integration Tests

```python
# tests/integration/test_champion_agent.py
def test_champion_agent_with_deepstack():
    """Test champion agent uses DeepStack correctly"""
    
def test_continual_resolving():
    """Test continual re-solving during gameplay"""
    
# tests/integration/test_training_pipeline.py
def test_data_generation():
    """Test training data generation"""
    
def test_model_training():
    """Test neural net training end-to-end"""
```

### 5.3 Validation Tests

Compare against known results from documentation:

```python
def test_tutorial_tree_exploitability():
    """Validate exploitability matches tutorial values.
    
    - Random strategy: ~175 chips
    - CFR (1000 iter): ~1.0 chips
    - DeepStack: ~1.37 chips
    """
    
def test_vs_reference_implementation():
    """Compare strategies against reference DeepStack"""
```

---

## 6. Implementation Phases

### Phase 1: Foundation (Week 1)
- ✅ Complete this blueprint
- Port core data structures (nodes, ranges, actions)
- Port TreeBuilder (complete implementation)
- Port TerminalEquity (complete implementation)
- Unit tests for foundation

### Phase 2: Core Solver (Week 2)
- Port TreeCFR (complete implementation)
- Port Lookahead (tensor-based CFR)
- Port CFRDGadget
- Validate against tutorial examples

### Phase 3: Neural Net (Week 3)
- Port ValueNN
- Port NetBuilder  
- Port MaskedHuberLoss
- Implement data generation
- Implement training loop

### Phase 4: Re-solving API (Week 4)
- Port Resolving class
- Port TreeStrategyFilling
- Integration tests
- Validation against reference

### Phase 5: Agent Integration (Week 5)
- Refactor ChampionAgent
- Update training pipeline
- End-to-end testing
- Performance profiling

### Phase 6: Polish (Week 6)
- Remove obsolete code
- Documentation updates
- Code review and optimization
- Final validation

---

## 7. Success Criteria

### Functional Requirements
- ✅ All core modules ported and functional
- ✅ Unit tests passing with >90% coverage
- ✅ Integration tests passing
- ✅ Tutorial examples reproduce documented results
- ✅ ChampionAgent uses DeepStack for decisions
- ✅ Training pipeline generates and trains models

### Performance Requirements
- Exploitability on tutorial tree: ~1.37 chips (within 10%)
- Re-solving time: <2 seconds per decision (CPU)
- Training time: <24 hours for 100K samples (GPU)

### Code Quality Requirements
- All modules have comprehensive docstrings
- Type hints throughout
- Consistent with existing codebase style
- No obsolete/duplicate code remains

---

## 8. Risk Mitigation

### Technical Risks

1. **Tensor Dimension Complexity**
   - Risk: Incorrect tensor indexing causing subtle bugs
   - Mitigation: Extensive unit tests, validate against reference outputs

2. **Lua/Python Semantic Differences**
   - Risk: Off-by-one errors (Lua is 1-indexed)
   - Mitigation: Careful review, automated tests

3. **GPU/CPU Compatibility**
   - Risk: Code only works on GPU or CPU
   - Mitigation: Test both, use `.to(device)` pattern

4. **Performance Regression**
   - Risk: Python slower than Lua
   - Mitigation: Profile, optimize hotspots, consider JIT compilation

### Schedule Risks

1. **Underestimated Complexity**
   - Risk: Lookahead porting takes longer than expected
   - Mitigation: Break into smaller milestones, adjust scope

2. **Integration Challenges**
   - Risk: Existing code not compatible
   - Mitigation: Incremental integration, adapter pattern

---

## 9. Deliverables Checklist

- [ ] PORTING_BLUEPRINT.md (this document)
- [ ] src/deepstack/lookahead.py (complete)
- [ ] src/deepstack/resolving.py (complete)
- [ ] src/deepstack/tree_cfr.py (enhanced)
- [ ] src/deepstack/value_nn.py (new)
- [ ] src/deepstack/tree_builder.py (enhanced)
- [ ] src/deepstack/tree_strategy_filling.py (new)
- [ ] src/deepstack/terminal_equity.py (enhanced)
- [ ] src/deepstack/cfrd_gadget.py (enhanced)
- [ ] src/deepstack/masked_huber_loss.py (enhanced)
- [ ] src/deepstack/data_generation.py (new)
- [ ] src/deepstack/trainer.py (new)
- [ ] src/agents/champion_agent.py (refactored)
- [ ] scripts/train_deepstack.py (new)
- [ ] tests/deepstack/*.py (comprehensive)
- [ ] tests/integration/*.py (comprehensive)
- [ ] README.md (updated)
- [ ] docs/DEEPSTACK_GUIDE.md (new)

---

## 10. References

### Documentation
- `data/doc/manual/tutorial.md` - Tutorial and examples
- `data/doc/manual/internals.md` - Internal architecture
- `data/doc/classes/*.html` - Class documentation
- `data/doc/modules/*.html` - Module documentation

### Papers
- "DeepStack: Expert-level artificial intelligence in heads-up no-limit poker" (Science, 2017)
- Original DeepStack repository: https://github.com/lifrordi/DeepStack-Leduc

### Existing Code
- `src/agents/champion_agent.py` - Current integration point
- `src/deepstack/*.py` - Stub implementations to enhance

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-14  
**Author**: Autonomous Coding Agent
