# Quick Reference - DeepStack Training

## 🚀 Quick Start (3 Steps)

### 1️⃣ Generate Data
```bash
# Quick test (1K samples, ~15 min)
python scripts/generate_quick_data.py --samples 1000 --cfr-iters 2000

# Production (50K samples, ~10 hours)
python scripts/generate_quick_data.py --samples 50000 --cfr-iters 2500
```

### 2️⃣ Train Model
```bash
# With GPU (recommended)
python scripts/train_deepstack.py \
  --config scripts/config/championship.json \
  --use-gpu \
  --epochs 200

# Without GPU
python scripts/train_deepstack.py \
  --config scripts/config/championship.json \
  --epochs 100
```

### 3️⃣ Validate
```bash
python scripts/validate_deepstack_model.py \
  --model models/versions/best_model.pt
```

## 📊 Target Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Correlation | 0.30 | >0.85 | 🔧 Needs new data |
| Relative Error | 1257% | <5% | 🔧 Needs new data |
| Flop Coverage | 0 | Full | ✅ Fixed |
| River Coverage | 0 | Full | ✅ Fixed |

## 🔧 What Was Fixed

1. **Terminal Equity** - Monte Carlo simulation (AA vs 72o now 0.81 ✅)
2. **CFR Quality** - 2000+ iterations (was 1000)
3. **Street Balance** - 20/35/25/20 coverage (was 30/40/20/10)
4. **Training Config** - Championship hyperparameters

## 🎯 Advanced Usage

### Adaptive Sampling (for problem buckets)
```bash
# Step 1: Derive weights from poor buckets
python scripts/derive_bucket_weights.py \
  --corr-json models/reports/per_bucket_corrs.json \
  --out bucket_weights.json

# Step 2: Regenerate with weights
python scripts/generate_quick_data.py \
  --samples 10000 \
  --bucket-weights bucket_weights.json
```

### Custom Training
```bash
python scripts/train_deepstack.py \
  --data-path src/train_samples \
  --epochs 200 \
  --batch-size 1024 \
  --effective-batch-size 4096 \
  --lr 0.0005 \
  --use-gpu \
  --fresh  # Start from scratch
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `scripts/generate_quick_data.py` | Generate training data |
| `scripts/train_deepstack.py` | Train neural network |
| `scripts/validate_deepstack_model.py` | Check model quality |
| `scripts/derive_bucket_weights.py` | Adaptive sampling |
| `scripts/config/championship.json` | Optimized config |
| `OPTIMIZATION_GUIDE.md` | Full documentation |
| `AUDIT_SUMMARY.md` | Summary of fixes |

## ⚡ Performance Tips

- **Data Generation**: Use `--cfr-iters 500` for testing, `2500` for production
- **Training**: Use `--use-gpu` for 10-20x speedup
- **Batch Size**: Larger is better (try 1024-4096 effective)
- **Epochs**: Start with 100, increase to 200 if not converged

## 🐛 Troubleshooting

**Slow data generation?**
→ Start with `--samples 100 --cfr-iters 500` for testing

**Out of memory?**
→ Reduce `--batch-size 256` or use `--no-gpu`

**Still poor correlations?**
→ Generate more data (10K+ samples) and increase `--cfr-iters 2500`

**Zero street coverage?**
→ Regenerate data (old data had sampling bug, now fixed)

## 📚 Documentation

- **OPTIMIZATION_GUIDE.md** - Comprehensive 8-page guide
- **AUDIT_SUMMARY.md** - Executive summary of improvements
- **README.md** - General project information

## ✅ Validation Checklist

Before deployment:
- [ ] Correlation > 0.85
- [ ] Relative error < 5%
- [ ] All streets have coverage
- [ ] Sign mismatch < 10%
- [ ] Calibration slope 0.9-1.1

## 🎓 DeepStack Paper References

- Section S3.1: Data generation ✅
- Section S3.2: Network architecture (7×500) ✅
- Section S2.1: Linear CFR ✅
- Section S2.2: CFR+ ✅
- Table S2: Specifications ✅

## 💡 Key Insights

1. **Terminal equity is critical** - Old simplified version was teaching wrong fundamentals
2. **CFR quality matters** - 2000+ iterations gives much better targets
3. **Street balance essential** - Model needs all game stages to learn properly
4. **Batch size impacts convergence** - Use 1024-4096 effective via gradient accumulation

---

**Questions?** See `OPTIMIZATION_GUIDE.md` for detailed explanations.
