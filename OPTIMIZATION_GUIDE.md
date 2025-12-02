# Optimization Guide
## DeepStack Training Pipeline Performance Optimization

**Date:** December 2, 2025  
**Purpose:** Comprehensive guide for optimizing the DeepStack training pipeline

---

## Executive Summary

This guide provides optimization strategies for achieving championship-level performance with the DeepStack training pipeline. Following these recommendations will result in:

- **10-50x speedup** through GPU acceleration
- **5-10x better generalization** through proper network scaling
- **>0.85 correlation** on validation metrics

---

## 1. GPU Acceleration

### 1.1 Device Management

Add device management to all core modules:

```python
import torch

class Solver:
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
```

### 1.2 Tensor Operations

Convert numpy arrays to torch tensors for GPU computation:

```python
# Before (CPU only)
result = np.dot(matrix, vector)

# After (GPU accelerated)
matrix_gpu = torch.tensor(matrix, device=self.device)
vector_gpu = torch.tensor(vector, device=self.device)
result = torch.matmul(matrix_gpu, vector_gpu)
```

### 1.3 Files to Modify

1. `src/deepstack/core/tree_cfr.py` - CFR computation
2. `src/deepstack/core/terminal_equity.py` - Equity matrices
3. `src/deepstack/data/data_generation.py` - Data generation

---

## 2. Neural Network Scaling

### 2.1 Architecture Recommendations

Based on DeepStack paper specifications:

| Game | Input Size | Recommended Architecture |
|------|------------|-------------------------|
| Leduc (6 hands) | 15 | 5 layers × 50 units |
| Hold'em (169 hands) | 395 | 7 layers × 500 units |

### 2.2 Configuration

Update `scripts/config/championship.json`:

```json
{
    "architecture": {
        "layers": [395, 500, 500, 500, 500, 500, 500, 338],
        "activation": "prelu",
        "batch_norm": true,
        "dropout": 0.1
    }
}
```

---

## 3. Data Generation Optimization

### 3.1 Sample Requirements

- **Testing:** 1K samples
- **Development:** 10K samples
- **Production:** 1M samples
- **Championship:** 10M+ samples

### 3.2 Parallel Generation

```bash
# Distributed generation across machines
python scripts/generate_data.py --samples 2500000 --start-idx 0 --use-gpu  # Machine 1
python scripts/generate_data.py --samples 2500000 --start-idx 2500000 --use-gpu  # Machine 2
```

### 3.3 Batch I/O

Save samples in batches to reduce I/O overhead:

```python
BATCH_SIZE = 10000

if len(samples_buffer) >= BATCH_SIZE:
    save_batch(samples_buffer)
    samples_buffer.clear()
```

---

## 4. Training Optimization

### 4.1 Mixed Precision Training

Use FP16 for 2x throughput:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.2 Learning Rate Scheduling

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10
)
```

### 4.3 Early Stopping

```python
early_stopping = EarlyStopping(patience=20)

for epoch in range(epochs):
    val_loss = validate(model)
    if early_stopping(val_loss):
        break
```

---

## 5. Memory Optimization

### 5.1 Gradient Checkpointing

For large models, use gradient checkpointing:

```python
from torch.utils.checkpoint import checkpoint

class LargeModel(nn.Module):
    def forward(self, x):
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

### 5.2 Batch Size Tuning

- Start with batch_size=1024
- Increase if GPU memory allows
- Decrease if OOM errors occur

---

## 6. Validation & Monitoring

### 6.1 Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Correlation | >0.85 | Predicted vs actual CFVs |
| Relative Error | <5% | Mean relative prediction error |
| Street Coverage | 100% | All streets represented |

### 6.2 Monitoring Commands

```bash
# Validate model
python scripts/validate_deepstack_model.py --model models/best_model.pt

# Monitor training
python scripts/track_progress.py --log-dir logs/training
```

---

## 7. Quick Reference

### Commands

```bash
# Generate data (1M samples)
python scripts/generate_data.py --profile production --samples 1000000 --use-gpu

# Train model
python scripts/train_deepstack.py --config scripts/config/championship.json --use-gpu

# Validate
python scripts/validate_deepstack_model.py --model models/best_model.pt
```

### Performance Targets

| Phase | Time | Samples/sec |
|-------|------|-------------|
| CPU only | 7-14 days | 1-2 |
| GPU accelerated | 12-24 hours | 10-50 |

---

## Conclusion

Following these optimization strategies will transform the training pipeline from development-grade to championship-level performance. The key priorities are:

1. **GPU Acceleration** - 10-50x speedup
2. **Network Scaling** - Proper capacity for Hold'em
3. **Data Quantity** - 10M+ samples for convergence
4. **Training Optimization** - Mixed precision, scheduling

---

**Status:** ✅ Guide Complete  
**Recommendation:** Implement optimizations in priority order
