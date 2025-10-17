# Data Generation Guide

## Quick Start

Generate training data quickly for development and testing:

```bash
# Small dataset for testing (fast - completes in seconds)
python scripts/generate_quick_data.py --samples 100 --cfr-iters 500

# Medium dataset for development (moderate - completes in minutes)
python scripts/generate_quick_data.py --samples 5000 --cfr-iters 1500

# Large dataset for production (slow - may take hours)
python scripts/generate_quick_data.py --samples 50000 --cfr-iters 2500
```

## Performance Optimizations

### Issue: Slow Data Generation

The original implementation had a critical performance issue that caused it to hang or run extremely slowly:

**Problem:** 
- `TerminalEquity` was recreated for each sample, recomputing expensive equity matrices
- Monte Carlo equity calculation did 169×169×500 = 14 million+ hand evaluations per sample
- No progress bars made it appear hung when it was just very slow

**Solution:**
1. **Reuse TerminalEquity instance** - Created once and shared across all samples
2. **Fast equity approximation** - Use rank-based approximation instead of Monte Carlo (100x speedup)
3. **Progress bars** - Added tqdm progress tracking for visibility

### Performance Comparison

| Configuration | Speed | Accuracy | Use Case |
|--------------|-------|----------|----------|
| Fast approximation (default) | ~10-20 samples/sec | Good | Quick testing, development |
| Monte Carlo (10 trials) | ~2-5 samples/sec | Better | Moderate accuracy needs |
| Monte Carlo (500+ trials) | ~0.02 samples/sec | Best | Championship-level (use pre-computed tables) |

### For Championship-Level Performance

To achieve championship-level accuracy similar to the original DeepStack paper:

1. **Pre-compute equity tables** - Run equity computation once offline and save to disk
2. **Use full Monte Carlo** - Set `fast_approximation=False` in TerminalEquity
3. **More samples** - Generate 10M+ samples as recommended in the paper
4. **More CFR iterations** - Use 2000+ CFR iterations per sample

Example for production:
```python
# In data_generation.py, modify ImprovedDataGenerator.__init__:
self.terminal_equity = TerminalEquity(
    game_variant='holdem', 
    num_hands=num_hands, 
    fast_approximation=False  # Use full Monte Carlo
)

# Then generate large dataset
python scripts/generate_quick_data.py --samples 1000000 --cfr-iters 2000
```

## Understanding the Trade-offs

### Fast Approximation (Default)
- **Pros:** 100x faster, good for rapid iteration
- **Cons:** Less accurate equity calculations
- **When to use:** Development, testing, initial training

### Full Monte Carlo
- **Pros:** Accurate equity calculations, championship-level
- **Cons:** Very slow without pre-computed tables
- **When to use:** Final production model training

## Troubleshooting

### Script appears to hang
- **Solution:** Install tqdm for progress bars: `pip install tqdm`
- The script may be computing equity matrices (takes 1-2 seconds with fast approximation)

### Out of memory
- **Solution:** Reduce `--samples` or generate in batches
- Each sample uses ~395 floats for input and ~338 for output

### Slow generation even with fast approximation
- **Check:** CFR iterations - reduce `--cfr-iters` for faster generation
- **Check:** Validate your CPU isn't thermal throttling
- **Expected:** ~10-20 samples/sec with CFR iterations=500

## Next Steps

After generating data:
1. Validate data quality: `python scripts/validate_data.py`
2. Train model: `python scripts/train.py`
3. Evaluate model: `python scripts/evaluate.py`
