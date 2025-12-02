# 🃏 Poker AI Championship Pipeline

**End-to-End DeepStack Implementation for Google Colab**

A complete, plug-and-play poker AI pipeline implementing the official DeepStack architecture. Optimized for Google Colab Free Tier (T4 GPU) with Unsloth-accelerated training.

## 🚀 3-Step Quick Start

### Step 1: Open in Colab
1. Upload this folder to your Google Drive under `My Drive/poker_ai`
2. Open `notebooks/inference.ipynb` in Google Colab
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)

### Step 2: Configure Paths
Edit the first cell of any notebook to point to your data:
```python
CONFIG = {
    "perception_model": "/content/drive/MyDrive/poker_ai/models/qwen_poker_vl",
    "decision_model": "/content/drive/MyDrive/poker_ai/models/deepstack_champion.pt",
    "coco_data": "/content/drive/MyDrive/poker_ai/data/annotations.coco.json"
}
```

### Step 3: Run!
Click **Runtime → Run all** in any notebook. That's it!

---

## 📁 Package Structure

```
poker_ai_colab_pipeline/
├── README.md                 # This file
├── config.yaml               # Central configuration
├── requirements.txt          # Colab-compatible dependencies
│
├── notebooks/
│   ├── inference.ipynb       # 🎮 End-to-end gameplay
│   ├── train_perception.ipynb    # 👁️ QWEN VL fine-tuning
│   ├── train_decision.ipynb      # 🧠 Value network training
│   └── validate_pipeline.ipynb   # ✅ Full validation suite
│
└── src/
    ├── perception/           # Vision inference pipeline
    ├── decision/             # DeepStack CFR engine
    ├── orchestration/        # Unified workflow
    └── validation/           # Compliance testing
```

---

## 📓 Notebooks

### 🎮 `inference.ipynb` - End-to-End Gameplay
Full pipeline: Screenshot → Game State → Optimal Action

**Features:**
- Automatic model loading from Google Drive
- QWEN 2.5-7B VL perception (4-bit quantized)
- DeepStack continual re-solving (2000 CFR iterations)
- Sub-second decision making on T4 GPU

**Usage:**
```python
from pipeline import PokerWorkflowOrchestrator

orchestrator = PokerWorkflowOrchestrator(
    perception_model="drive/models/qwen_vl_final",
    decision_model="drive/models/deepstack_champion.pt"
)

result = orchestrator.process_screenshot("screenshot.png")
print(f"Action: {result.action}, Amount: {result.amount}")
```

### 👁️ `train_perception.ipynb` - Vision Model Training
Fine-tune QWEN 2.5-7B VL on poker screenshots

**Features:**
- Unsloth 4-bit LoRA fine-tuning
- Automatic COCO dataset loading
- Gradient checkpointing for 16GB VRAM
- Drive checkpoint saving every 100 steps

**Requirements:**
- COCO-formatted annotations (`annotations.coco.json`)
- Image directory with poker screenshots

### 🧠 `train_decision.ipynb` - Value Network Training
Train DeepStack-style value network

**Features:**
- Pre-generated CFR sample loading
- Temperature scaling calibration
- Early stopping with patience
- ONNX export with quantization

**Requirements:**
- CFR-solved training samples (`.npz` format)

### ✅ `validate_pipeline.ipynb` - Full Validation
Comprehensive compliance testing

**Tests:**
- DeepStack Leduc fidelity (strategy comparison)
- Value network calibration (ECE < 0.05)
- Perception accuracy (>98%)
- Memory stability (100+ cycles)

---

## ⚙️ Configuration

Edit `config.yaml` for your setup:

```yaml
drive:
  base_path: /content/drive/MyDrive/poker_ai
  models:
    perception: ${drive.base_path}/models/qwen_poker_vl
    decision: ${drive.base_path}/models/deepstack_champion.pt

perception:
  quantization:
    enabled: true
    bits: 4

decision:
  cfr:
    iterations: 2000
    lookahead_depth: 3
```

---

## 🎯 DeepStack Compliance

This implementation follows the official DeepStack architecture:

| Component | Specification | Status |
|-----------|--------------|--------|
| **Offline Training** | CFR-solved poker situations → neural network | ✅ |
| **Online Re-solving** | Depth-limited lookahead + neural terminal values | ✅ |
| **Input Format** | `node_params` + normalized range vectors | ✅ |
| **Output** | Counterfactual values for all private hands | ✅ |
| **Strategy** | Nash equilibrium approximation | ✅ |

---

## 📊 Expected Results

| Metric | Target | Notes |
|--------|--------|-------|
| Perception Accuracy | >98% | COCO validation set |
| Value Network ECE | <0.05 | Temperature-scaled |
| Strategy Correlation | >0.85 | vs DeepStack-Leduc |
| Inference Latency | <1s | T4 GPU |
| VRAM Usage | <14GB | With quantization |

---

## 🔧 Troubleshooting

### "CUDA out of memory"
- Enable gradient checkpointing in config
- Reduce batch size to 1
- Clear cache: `torch.cuda.empty_cache()`

### "Model not found"
- Verify Drive is mounted: `drive.mount('/content/drive')`
- Check path in config matches your Drive structure

### "Low accuracy"
- Increase CFR iterations (2000+ recommended)
- Ensure sufficient training samples (10K+ for development)

---

## 📚 References

- [DeepStack Science Paper](https://www.science.org/doi/10.1126/science.aam6960) (Moravčík et al., 2017)
- [DeepStack-Leduc GitHub](https://github.com/lifrordi/DeepStack-Leduc)
- [Unsloth](https://github.com/unslothai/unsloth) - Memory-efficient fine-tuning

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built for Championship-Grade Poker AI** 🏆
