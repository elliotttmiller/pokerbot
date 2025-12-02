# Perception Model (Croupier)

This directory contains the fine-tuned VLM model artifacts for game state perception.

## Expected Contents
- `config.json` - Model configuration
- `model.safetensors` - Model weights (or `pytorch_model.bin`)
- `preprocessor_config.json` - Preprocessor configuration
- `tokenizer.json` - Tokenizer configuration

## Model
- Base: QWEN 2.5-VL (7B)
- Task: Poker game state extraction from screenshots
- Output: Structured JSON with hole cards, community cards, pot, etc.

## Training
See `docs/TRAINING_GUIDE.md` for model training instructions.
