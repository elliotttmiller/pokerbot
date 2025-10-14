"""
Automated batch size and model architecture optimization for DeepStack value network.
Finds optimal batch size and model configuration for best speed and accuracy.
"""
import time
import numpy as np
import tensorflow as tf
from src.agents.champion_agent import load_deepstack_train_samples, train_value_network_on_deepstack_samples

def optimize_batch_and_model(samples_dir='data/train_samples', batch_sizes=[16,32,64], layer_sizes=[32,64,128]):
    samples = load_deepstack_train_samples(samples_dir)
    best_val_loss = float('inf')
    best_config = None
    for batch_size in batch_sizes:
        for layer_size in layer_sizes:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=(27,)),
                tf.keras.layers.Dense(layer_size, activation='relu'),
                tf.keras.layers.Dense(13)
            ])
            start = time.time()
            history = train_value_network_on_deepstack_samples(model, samples, epochs=3, batch_size=batch_size)
            end = time.time()
            val_loss = history.history['val_loss'][-1]
            print(f"Batch: {batch_size}, Layer: {layer_size}, Time: {end-start:.2f}s, Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = (batch_size, layer_size)
    print(f"Best config: Batch size={best_config[0]}, Layer size={best_config[1]}, Val Loss={best_val_loss:.4f}")
    return best_config

if __name__ == '__main__':
    optimize_batch_and_model()
