# Champion Agent Training Pipeline - Quick Start

## 🚀 Quick Start (5 Minutes)

Run a quick smoketest to verify everything works:

```bash
# Navigate to project root
cd /path/to/pokerbot

# Run smoketest training (5-7 minutes)
python scripts/train_champion.py --mode smoketest

# Validate the trained agent
python scripts/validate_training.py --model models/champion_final --hands 100
```

That's it! You now have a trained Champion Agent.

## 📖 Full Documentation

For complete documentation, see **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)** which includes:
- Detailed explanation of the training pipeline
- Full production training instructions
- Troubleshooting guide
- Advanced usage examples
- Step-by-step validation procedures

## 🎯 What You Get

After running the smoketest, you'll have:
- ✅ **Trained agent model** at `models/champion_final.cfr` and `models/champion_final.keras`
- ✅ **Training report** in `logs/training_report_*.txt`
- ✅ **Validation results** showing agent performance
- ✅ **Checkpoints** for each training stage

## 📊 Expected Performance (Smoketest)

Your agent should achieve approximately:
- **60%+ win rate** vs Random agents
- **50%+ win rate** vs CFR agents
- **Consistent decision making** (90%+ consistency score)

For better performance, run full training mode:
```bash
python scripts/train_champion.py --mode full
```

## 🔧 Requirements

- Python 3.8+
- NumPy
- TensorFlow
- 4GB RAM
- 2GB disk space

## 📁 Project Structure

```
scripts/
  ├── train_champion.py       # Main training pipeline
  └── validate_training.py    # Validation script
  
docs/
  └── TRAINING_GUIDE.md       # Complete user guide
  
models/                       # Saved models
logs/                         # Training logs
```

## 🎓 Training Stages

The pipeline trains your agent through 3 progressive stages:

1. **Stage 1: CFR Warmup** - Game-theoretic foundation
2. **Stage 2: Self-Play** - Iterative improvement
3. **Stage 3: Vicarious Learning** - Learn from diverse opponents

## 💡 Tips

- Always start with `--mode smoketest` to verify setup
- Use `--episodes N` to customize training length
- Check `logs/` for detailed training reports
- Run validation to verify agent quality

## ❓ Need Help?

See the comprehensive guide: **[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)**
