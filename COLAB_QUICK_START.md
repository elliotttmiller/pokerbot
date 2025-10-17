# Quick Reference: Using pokerbot_colab.ipynb

## 🚀 Getting Started (3 Simple Steps)

### Step 1: Open in Google Colab
1. Go to https://colab.research.google.com/
2. File → Open Notebook → GitHub tab
3. Enter: `elliotttmiller/pokerbot`
4. Select: `pokerbot_colab.ipynb`

### Step 2: Enable GPU
1. Runtime → Change runtime type
2. Hardware accelerator: **GPU** (select T4 or better)
3. Click Save

### Step 3: Run All Cells
1. Runtime → Run all
2. Or press `Ctrl+F9` (Windows/Linux) or `Cmd+F9` (Mac)
3. Watch the progress in each cell

## 📊 What Each Cell Does

**Code Cells (execute these):**

| Cell | What It Does | Time | Can Skip? |
|------|-------------|------|-----------|
| 1 | Clone repository | 10s | No |
| 2 | Install dependencies | 2-3m | No |
| 3 | Check GPU & system | 5s | No |
| 4 | Generate training data | 15-20m | No |
| 5 | Train model | 30-40m | No |
| 6 | Validate results | 1-2m | No |
| 8 | Generate reports | 2-3m | Yes* |
| 10 | Interactive viz | 1m | Yes* |
| 11 | Download model | 30s | Yes* |

**Documentation Cells (markdown - read these):**
- Cell 0: Introduction and setup
- Cell 7: Results analysis guide
- Cell 9: Advanced features
- Cell 12: Troubleshooting guide

*Optional but recommended

## ⏱️ Total Time Estimates

- **Quick Test** (100 samples): ~30 minutes
- **Standard Training** (1K samples): ~1-2 hours
- **Production** (50K samples): ~10-12 hours

## 🎯 Expected Results

### Good Results ✅
- Correlation: > 0.85
- Relative Error: < 5%
- All streets have data

### Needs Work ⚠️
- Correlation: 0.5 - 0.85
- Relative Error: 5% - 20%

### Poor Results ❌
- Correlation: < 0.5
- Relative Error: > 20%

## 🔧 Common Modifications

### Quick Test (faster, less accurate)
**Cell 4:**
```python
!python scripts/generate_quick_data.py --samples 100 --cfr-iters 500
```

**Cell 5:**
```python
!python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu --epochs 50
```

### Production Training (slower, best accuracy)
**Cell 4:**
```python
!python scripts/generate_quick_data.py --samples 50000 --cfr-iters 2500
```

**Cell 5:**
```python
!python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu --epochs 300
```

## 🐛 Troubleshooting

### "No GPU detected"
→ Go to Runtime → Change runtime type → Select GPU

### "Out of memory"
→ Reduce batch size or use fewer samples

### "Training too slow"
→ Check Cell 3 output - make sure GPU is enabled

### "Poor correlation results"
→ Generate more data (increase `--samples`)
→ Train longer (increase `--epochs`)

### "Cell failed to execute"
→ Check error message in cell output
→ Restart runtime: Runtime → Restart runtime
→ Run cells in order from top

### "Can't download files"
→ Check if model file exists: `models/versions/best_model.pt`
→ Try manual download from Files panel (left sidebar)

## 💾 Where Files Are Saved

```
/content/pokerbot/
├── models/
│   ├── versions/
│   │   └── best_model.pt         ← Your trained model
│   └── reports/                  ← Analysis plots & metrics
├── src/
│   └── train_samples/            ← Generated training data
└── scripts/
    └── config/
        └── championship.json     ← Training configuration
```

## 📥 Downloading Your Model

**Option 1: Use Cell 11** (Recommended)
- Automatically zips everything
- Includes model, reports, and config
- Downloads to your computer

**Option 2: Manual Download**
1. Click Files icon (left sidebar)
2. Navigate to `models/versions/`
3. Right-click `best_model.pt`
4. Select "Download"

## 🔄 Re-running the Notebook

The notebook is designed to be safely re-run:
- ✅ Won't re-clone if repo exists
- ✅ Updates to latest code version
- ✅ Skips if dependencies installed
- ✅ Can restart from any cell

**To start fresh:**
1. Runtime → Restart runtime
2. Runtime → Run all

## 📚 Additional Resources

- **Full optimization guide**: See `COLAB_NOTEBOOK_OPTIMIZATION.md`
- **Training guide**: See `QUICK_REFERENCE.md`
- **Detailed docs**: See `OPTIMIZATION_GUIDE.md`
- **GitHub repo**: https://github.com/elliotttmiller/pokerbot

## 🆘 Getting Help

1. **Check the troubleshooting guide** (last markdown cell in notebook)
2. **Review error messages** in failed cells
3. **Check documentation** in repository
4. **Open an issue** on GitHub if stuck

## 💡 Pro Tips

1. **Start small**: Try 100-1000 samples first
2. **Use GPU**: 10-20x faster than CPU
3. **Monitor progress**: Watch cell outputs
4. **Save your work**: Download model before closing
5. **Check metrics**: Review validation results before scaling up

## ⚙️ Advanced Usage

### Custom Training Config
Edit Cell 5 to use different config:
```python
# Use a different pre-configured training setup
!python scripts/train_deepstack.py --config scripts/config/training.json --use-gpu

# Available configs:
# - championship.json (default, best for production)
# - training.json (alternative balanced config)
# - smoketest.json (quick test, minimal training)
```

### Adaptive Bucket Weighting
After initial training with poor results:
```python
# Derive weights from problem buckets
!python scripts/derive_bucket_weights.py \
  --corr-json models/reports/per_bucket_corrs.json \
  --out bucket_weights.json

# Regenerate with weights
!python scripts/generate_quick_data.py \
  --samples 10000 \
  --bucket-weights bucket_weights.json
```

### Resume Training
If training interrupted:
```python
!python scripts/train_deepstack.py \
  --config scripts/config/championship.json \
  --use-gpu \
  --resume  # Continues from last checkpoint
```

---

**Happy Training! 🎰🤖**

*Last updated: 2025-10-17*
