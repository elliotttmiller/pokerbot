#!/usr/bin/env python3
"""CLI for pruning and quantizing Keras models.

Examples:
  python scripts/prune_quantize_model.py --input models/versions/champion_best.keras --output models/versions/champion_best_pruned.keras --prune --sparsity 0.6
  python scripts/prune_quantize_model.py --input models/versions/champion_best.keras --tflite models/versions/champion_best.tflite --quantize
"""
import argparse
import os
import tensorflow as tf

from src.utils.model_optimization import prune_model, strip_pruning, quantize_model, save_tflite


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to Keras model (.keras or .h5)")
    p.add_argument("--output", default=None, help="Path to save pruned model")
    p.add_argument("--tflite", default=None, help="Path to save TFLite model")
    p.add_argument("--prune", action="store_true")
    p.add_argument("--quantize", action="store_true")
    p.add_argument("--sparsity", type=float, default=0.5)
    args = p.parse_args()

    model = tf.keras.models.load_model(args.input, compile=False)

    if args.prune:
        model = prune_model(model, final_sparsity=args.sparsity)
        model = strip_pruning(model)
        if args.output:
            model.save(args.output)
            print(f"Pruned model saved to {args.output}")

    if args.quantize:
        tflite = quantize_model(model)
        if args.tflite:
            save_tflite(tflite, args.tflite)
            print(f"Quantized TFLite model saved to {args.tflite}")


if __name__ == "__main__":
    main()
