# Championship-Level Training - Implementation Checklist

## 🎯 Your Mission: Transform Model Performance from 0.33 to >0.85 Correlation

Current problem: Only 5,000 training samples (100x too small)
Solution: Generate 50K-500K samples with championship settings

---

## Week 1: Quick Fix ⚡ (Achieve 0.65-0.75 Correlation)

### Day 1-2: Generate Data (7-10 hours runtime)
- [ ] Run production data generation
  ```bash
  python scripts/generate_production_data.py --samples 50000 --cfr-iters 2000
  ```
- [ ] Let it run overnight or over weekend
- [ ] Expected time: 7-10 hours
- [ ] Expected output: `src/train_samples_production/` with 50K samples

### Day 3: Train Model (2-4 hours runtime)
- [ ] Train with championship config
  ```bash
  python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu --epochs 200
  ```
- [ ] Monitor training progress
- [ ] Wait for early stopping or 200 epochs
- [ ] Expected time: 2-4 hours (GPU) or 8-12 hours (CPU)

### Day 4: Validate Results
- [ ] Run validation script
  ```bash
  python scripts/validate_deepstack_model.py
  ```
- [ ] Check key metrics:
  - [ ] Correlation: should be 0.65-0.75 (vs current 0.33)
  - [ ] Relative Error: should be 10-20% (vs current 1500%)
  - [ ] Sign Mismatch: should be <15% (vs current 23%)
  - [ ] Per-street correlation: all should be positive

### Week 1 Success Criteria
- ✅ Correlation: 0.65-0.75
- ✅ Model usable for development/testing
- ✅ Clear path to further improvement

---

## Week 2-3: Production Quality 🚀 (Achieve 0.75-0.85 Correlation)

### Week 2: Generate Larger Dataset (18-24 hours runtime)
- [ ] Run production data generation with more samples
  ```bash
  python scripts/generate_production_data.py --samples 100000 --cfr-iters 2500
  ```
- [ ] Expected time: 18-24 hours
- [ ] Can run over weekend
- [ ] Expected output: 100K samples

### Week 3: Train & Validate
- [ ] Train with championship config
  ```bash
  python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu --epochs 200
  ```
- [ ] Validate results
- [ ] If per-bucket correlations show weak buckets:
  - [ ] Run bucket weight derivation
    ```bash
    python scripts/derive_bucket_weights.py --corr-json models/reports/per_bucket_corrs.json --out bucket_weights.json
    ```
  - [ ] Regenerate data with bucket weights
    ```bash
    python scripts/generate_production_data.py --samples 100000 --bucket-weights bucket_weights.json
    ```

### Production Success Criteria
- ✅ Correlation: 0.75-0.85
- ✅ Model ready for real play
- ✅ Sign mismatch <10%

---

## Long-Term: Championship Level 🏆 (Achieve >0.85 Correlation)

### Month 1: Generate Championship Dataset (4-5 days runtime)
- [ ] Plan multi-day data generation
- [ ] Set up tmux/screen for persistent session
  ```bash
  tmux new -s championship
  python scripts/generate_production_data.py --samples 500000 --cfr-iters 2500 --adaptive-cfr
  # Ctrl+B, D to detach
  ```
- [ ] Expected time: 4-5 days
- [ ] Monitor progress periodically
  ```bash
  tmux attach -t championship
  ```

### Month 2: Train to Convergence
- [ ] Train with championship config
- [ ] Use GPU for reasonable training time
- [ ] Monitor validation metrics
- [ ] May need 200+ epochs

### Championship Success Criteria
- ✅ Correlation: >0.85
- ✅ Relative Error: <5%
- ✅ Sign Mismatch: <10%
- ✅ All per-street correlations >0.6
- ✅ Calibration slope 0.9-1.1
- ✅ Ready for serious competition

---

## Daily Checklist (During Data Generation)

### Morning
- [ ] Check data generation progress
  - [ ] Look for progress bars in terminal
  - [ ] Verify samples are being generated
  - [ ] Check no errors in output

### Evening
- [ ] Estimate time remaining
  - [ ] Calculate: (total_samples - current) / samples_per_sec
  - [ ] Plan when to start training

### If Problems Occur
- [ ] Check disk space: `df -h`
- [ ] Check memory: `free -h` or `top`
- [ ] Check CPU usage: should be near 100%
- [ ] If errors, check logs for specific issues

---

## Resource Requirements

### Disk Space
- [ ] 5K samples: ~100 MB
- [ ] 50K samples: ~1 GB
- [ ] 100K samples: ~2 GB
- [ ] 500K samples: ~10 GB
- [ ] Ensure at least 2x the expected size available

### Memory (RAM)
- [ ] Data generation: 4-8 GB recommended
- [ ] Training 50K samples: 8-16 GB
- [ ] Training 100K samples: 16-32 GB
- [ ] Training 500K samples: 32-64 GB

### GPU (Optional but Recommended)
- [ ] Training 100K samples:
  - GPU: 2-4 hours
  - CPU: 8-12 hours
- [ ] Recommended: NVIDIA GPU with 6+ GB VRAM
- [ ] Can use Google Colab for free GPU

### Time Investment
- [ ] Quick fix (50K samples): 1 week
- [ ] Production (100K samples): 2-3 weeks
- [ ] Championship (500K samples): 1-2 months

---

## Troubleshooting Checklist

### Data Generation Slow
- [ ] Check CFR iterations (lower = faster)
- [ ] Try `--simple-bet-sizing` for speed
- [ ] Consider `--adaptive-cfr` for efficiency
- [ ] Verify CPU is at ~100% usage

### Out of Memory
- [ ] Generate in smaller batches
- [ ] Close other applications
- [ ] Consider cloud instance with more RAM

### Training Not Improving
- [ ] Verify data was generated successfully
  ```bash
  ls -lh src/train_samples/*.pt
  ```
- [ ] Check data quality
  ```bash
  python scripts/validate_data.py
  ```
- [ ] Ensure GPU is being used (check logs)
- [ ] Try lowering learning rate

### Model Still Poor After Training
- [ ] Check data quantity (should be 50K+ samples)
- [ ] Verify training completed (not interrupted)
- [ ] Check per-bucket correlations for weak spots
- [ ] Consider regenerating with bucket weights

---

## Success Metrics Tracker

### Initial State (Before Improvements)
- Samples: 5,000
- Correlation: 0.33
- Relative Error: 1503%
- Sign Mismatch: 23%
- Status: ❌ Unusable

### After Week 1 (50K samples)
- Samples: 50,000
- Correlation: _____ (target: 0.65-0.75)
- Relative Error: _____ (target: 10-20%)
- Sign Mismatch: _____ (target: <15%)
- Status: [ ] Acceptable

### After Week 3 (100K samples)
- Samples: 100,000
- Correlation: _____ (target: 0.75-0.85)
- Relative Error: _____ (target: 5-10%)
- Sign Mismatch: _____ (target: <10%)
- Status: [ ] Production-ready

### After Month 2 (500K samples)
- Samples: 500,000
- Correlation: _____ (target: >0.85)
- Relative Error: _____ (target: <5%)
- Sign Mismatch: _____ (target: <10%)
- Status: [ ] Championship-level

---

## Quick Reference Commands

### Generate Data
```bash
# Quick (50K samples, 7-10 hours)
python scripts/generate_production_data.py --samples 50000 --cfr-iters 2000

# Production (100K samples, 18-24 hours)
python scripts/generate_production_data.py --samples 100000 --cfr-iters 2500

# Championship (500K samples, 4-5 days)
python scripts/generate_production_data.py --samples 500000 --cfr-iters 2500 --adaptive-cfr
```

### Train
```bash
python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu --epochs 200
```

### Validate
```bash
python scripts/validate_deepstack_model.py
```

### Check Progress (in tmux)
```bash
tmux attach -t datagen  # or championship
```

### Derive Bucket Weights (if needed)
```bash
python scripts/derive_bucket_weights.py --corr-json models/reports/per_bucket_corrs.json --out bucket_weights.json
```

---

## Notes & Observations

Use this space to track your progress, observations, and any issues:

```
Date: _________
Action: _________
Result: _________
Notes: _________

Date: _________
Action: _________
Result: _________
Notes: _________

Date: _________
Action: _________
Result: _________
Notes: _________
```

---

## Final Checklist

Before deploying model:
- [ ] Correlation >0.75 (minimum for real play)
- [ ] Relative Error <10%
- [ ] Sign Mismatch <10%
- [ ] All per-street correlations positive
- [ ] Temperature scaling applied
- [ ] Model tested on validation set
- [ ] Performance documented
- [ ] Ready for deployment

---

**Remember:** The code is already championship-level. You just need patience to generate enough training data. The time investment is worth it for the performance improvement.

**Good luck!** 🚀
