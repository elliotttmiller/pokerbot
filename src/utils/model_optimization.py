"""Model pruning and quantization utilities for TensorFlow/Keras models.

Provides:
- prune_model: Apply magnitude-based pruning during fine-tuning
- quantize_model: Apply post-training dynamic range quantization (TFLite)
- save_tflite: Convert and save a Keras model to TFLite
"""
from __future__ import annotations

from typing import Optional


def prune_model(model, final_sparsity: float = 0.5, begin_step: int = 0, end_step: int = 1000):
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise RuntimeError("tensorflow-model-optimization not installed")

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step,
        )
    }
    return prune_low_magnitude(model, **pruning_params)


def strip_pruning(model):
    import tensorflow_model_optimization as tfmot
    return tfmot.sparsity.keras.strip_pruning(model)


def quantize_model(model):
    """Return a TFLite flatbuffer with dynamic range quantization."""
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model


def save_tflite(tflite_model: bytes, filepath: str):
    with open(filepath, 'wb') as f:
        f.write(tflite_model)
