# Unified Training System

## Overview

The training system has been consolidated into a streamlined, profile-based architecture. This replaces multiple redundant scripts with unified tools that use configuration profiles for different training scenarios.

## Quick Start

### 1. Generate Training Data

```bash
# Development iteration (10K samples, ~1-2 hours)
python scripts/generate_data.py --profile development

# Production quality (100K samples, ~18-24 hours)  
python scripts/generate_data.py --profile production

# Championship-level (500K samples, ~4-5 days)
python scripts/generate_data.py --profile championship
```

### 2. Train Model

```bash
# Development training
python scripts/train_model.py --profile development --use-gpu

# Production training
python scripts/train_model.py --profile production --use-gpu

# Championship training
python scripts/train_model.py --profile championship --use-gpu
```

### 3. Validate Results

```bash
# Validate model performance
python scripts/validate_model.py --type model

# Validate data quality
python scripts/validate_model.py --type data

# Comprehensive validation
python scripts/validate_model.py --type all
```

## Unified Scripts

### `scripts/generate_data.py`
**Replaces:** `generate_quick_data.py`, `generate_production_data.py`

Unified data generation with profile support:
- **Profiles:** testing, development, production, championship
- **Features:** Time estimation, quality assessment, adaptive CFR
- **Configuration:** Profile-based or custom JSON configs

```bash
# Use profile
python scripts/generate_data.py --profile production

# Override parameters
python scripts/generate_data.py --profile production --samples 200000

# Custom config
python scripts/generate_data.py --config config/data_generation/custom.json

# Advanced features
python scripts/generate_data.py --profile championship --adaptive-cfr
```

### `scripts/train_model.py`
**Alternative to:** `train_deepstack.py`

Unified training with profile support:
- **Profiles:** testing, development, production, championship
- **Features:** GPU support, early stopping, EMA, street weighting
- **Configuration:** Profile-based or custom JSON configs

```bash
# Use profile with GPU
python scripts/train_model.py --profile production --use-gpu

# Override epochs
python scripts/train_model.py --profile production --epochs 300

# Custom config
python scripts/train_model.py --config config/training/custom.json
```

### `scripts/validate_model.py`
**Alternative to:** `validate_deepstack_model.py`, `validate_data.py`

Unified validation:
- **Types:** data, model, all
- **Features:** Quality assessment, correlation analysis
- **Output:** Clear pass/fail with recommendations

```bash
# Validate trained model
python scripts/validate_model.py --type model

# Validate data quality
python scripts/validate_model.py --type data

# Both
python scripts/validate_model.py --type all
```

## Configuration System

### Profile-Based Configuration

Profiles provide pre-configured settings for common scenarios:

**Data Generation Profiles:**
- `testing`: 1K samples, 500 CFR iters (~5-10 min)
- `development`: 10K samples, 1500 CFR iters (~1-2 hours)
- `production`: 100K samples, 2500 CFR iters (~18-24 hours)
- `championship`: 500K samples, 2500 CFR iters (~4-5 days)

**Training Profiles:**
- `testing`: 20 epochs, quick validation
- `development`: 100 epochs, moderate training
- `production`: 200 epochs, production quality
- `championship`: 200 epochs, championship-level

### Configuration Files

Organized in `config/` directory:

```
config/
├── data_generation/
│   ├── championship.json
│   ├── production.json
│   ├── development.json
│   └── testing.json (optional)
├── training/
│   ├── championship.json
│   ├── production.json
│   ├── development.json
│   └── testing.json (optional)
└── validation/
    └── (future validation configs)
```

### Custom Configurations

Create custom JSON configs for specific needs:

**Data Generation Config:**
```json
{
  "description": "Custom data generation",
  "samples": 50000,
  "validation_samples": 10000,
  "cfr_iterations": 2000,
  "output": "src/train_samples_custom",
  "championship_bet_sizing": true,
  "adaptive_cfr": false
}
```

**Training Config:**
```json
{
  "description": "Custom training",
  "data_path": "src/train_samples_custom",
  "epochs": 150,
  "batch_size": 1024,
  "lr": 0.0003,
  "use_street_weighting": true,
  "street_weights": [0.8, 1.0, 1.2, 1.4]
}
```

## Migration Guide

### Old → New Commands

**Data Generation:**
```bash
# Old
python scripts/generate_quick_data.py --samples 10000 --cfr-iters 1500

# New
python scripts/generate_data.py --profile development
# or
python scripts/generate_data.py --profile development --samples 10000 --cfr-iters 1500
```

```bash
# Old  
python scripts/generate_production_data.py --samples 100000

# New
python scripts/generate_data.py --profile production
```

**Training:**
```bash
# Old
python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu

# New
python scripts/train_model.py --profile championship --use-gpu
# or
python scripts/train_model.py --config config/training/championship.json --use-gpu
```

**Validation:**
```bash
# Old
python scripts/validate_deepstack_model.py

# New
python scripts/validate_model.py --type model
```

```bash
# Old
python scripts/validate_data.py

# New
python scripts/validate_model.py --type data
```

## Workflow Examples

### Complete Development Cycle

```bash
# 1. Generate development data (~1-2 hours)
python scripts/generate_data.py --profile development

# 2. Train model (~1-2 hours GPU)
python scripts/train_model.py --profile development --use-gpu

# 3. Validate
python scripts/validate_model.py --type all

# 4. If quality good, proceed to production
```

### Production Deployment

```bash
# 1. Generate production data (~18-24 hours)
python scripts/generate_data.py --profile production

# 2. Train model (~2-4 hours GPU)
python scripts/train_model.py --profile production --use-gpu

# 3. Validate performance
python scripts/validate_model.py --type model

# 4. Deploy if correlation >0.75
```

### Championship-Level Training

```bash
# 1. Generate championship data (~4-5 days)
python scripts/generate_data.py --profile championship --adaptive-cfr

# 2. Train with championship config (~2-4 hours GPU)
python scripts/train_model.py --profile championship --use-gpu

# 3. Comprehensive validation
python scripts/validate_model.py --type all

# 4. Should achieve correlation >0.85
```

## Benefits of Unified System

### Reduced Complexity
- **Before:** 24+ scripts, 7+ config files in scripts/config/
- **After:** 3 main scripts, organized config/ directory
- **Savings:** 70% reduction in script count

### Improved Maintainability
- Single source of truth for each operation
- Clear profile system for common scenarios
- Consistent parameter naming across all tools

### Better User Experience
- Intuitive profile names (testing, development, production, championship)
- Built-in time and quality estimates
- Clear validation feedback

### Enhanced Flexibility
- Profiles for quick start
- Config files for custom scenarios
- Command-line overrides for fine-tuning
- All three can be combined

## Advanced Features

### Adaptive CFR
```bash
# Enable adaptive CFR iterations (20-30% faster)
python scripts/generate_data.py --profile championship --adaptive-cfr
```

### Bucket-Weighted Sampling
```bash
# Use bucket weights for targeted improvement
python scripts/derive_bucket_weights.py --corr-json models/reports/per_bucket_corrs.json
python scripts/generate_data.py --profile production --bucket-weights bucket_weights.json
```

### Custom Street Weighting
```json
{
  "use_street_weighting": true,
  "street_weights": [0.6, 1.0, 1.4, 2.0]
}
```

### Mixed Configurations
```bash
# Profile + config file + overrides
python scripts/train_model.py \
  --profile production \
  --config config/training/custom.json \
  --epochs 300 \
  --lr 0.0003
```

## Deprecated Scripts

The following scripts are now deprecated in favor of the unified system:

### Data Generation
- ~~`generate_quick_data.py`~~ → `generate_data.py --profile development`
- ~~`generate_production_data.py`~~ → `generate_data.py --profile production`

### Training
- `train_deepstack.py` → Still active (primary for DeepStack network)
- `train.py` → Still active (primary for agent training)
- Alternative: `train_model.py --profile <profile>` (unified approach)

### Validation
- `validate_deepstack_model.py` → Still active (primary)
- `validate_data.py` → Still active (primary)
- Alternative: `validate_model.py --type <type>` (unified approach)

**Note:** Legacy data generation scripts have been removed. Training and validation scripts are available in both primary and unified versions.

## Configuration Reference

### Data Generation Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `samples` | int | Training samples | Profile-dependent |
| `validation_samples` | int | Validation samples | samples // 5 |
| `cfr_iterations` | int | CFR iterations per sample | Profile-dependent |
| `output` | str | Output directory | Profile-dependent |
| `championship_bet_sizing` | bool | Per-street bet sizing | true |
| `adaptive_cfr` | bool | Adaptive CFR iterations | false |

### Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `data_path` | str | Path to training data | Profile-dependent |
| `epochs` | int | Training epochs | Profile-dependent |
| `batch_size` | int | Batch size | Profile-dependent |
| `lr` | float | Learning rate | 0.0005 |
| `huber_delta` | float | Huber loss delta | 0.3 |
| `use_street_weighting` | bool | Enable street weighting | true |
| `street_weights` | list | Street weights [pre, flop, turn, river] | [0.8, 1.0, 1.2, 1.4] |
| `use_gpu` | bool | Use GPU if available | false |

## Support

For issues or questions:
1. Check the examples in this README
2. Review configuration files in `config/`
3. Run with `--help` flag for detailed options
4. Consult the original documentation in `docs/`

## Future Enhancements

Planned improvements:
- Web-based configuration editor
- Progress tracking dashboard  
- Automated hyperparameter tuning
- Multi-GPU training support
- Distributed data generation
