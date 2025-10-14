"""
Performance profiling script for DeepStack value network training and inference.
Measures training time, inference latency, and resource usage.
"""
import time
import numpy as np
import tensorflow as tf
from src.agents.champion_agent import load_deepstack_train_samples, train_value_network_on_deepstack_samples

def profile_training(samples_dir='data/train_samples', epochs=3, batch_size=32):
    samples = load_deepstack_train_samples(samples_dir)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(13)
    ])
    start = time.time()
    history = train_value_network_on_deepstack_samples(model, samples, epochs=epochs, batch_size=batch_size)
    end = time.time()
    print(f"Training time: {end-start:.2f} seconds")
    print(f"Final loss: {history.history['loss'][-1]:.4f}, Final val_loss: {history.history['val_loss'][-1]:.4f}")
    return model

def profile_inference(model, num_samples=100):
    dummy_inputs = np.random.rand(num_samples, 27).astype(np.float32)
    start = time.time()
    _ = model.predict(dummy_inputs, batch_size=32, verbose=0)
    end = time.time()
    print(f"Inference time for {num_samples} samples: {end-start:.2f} seconds")
    print(f"Average latency per sample: {(end-start)/num_samples*1000:.2f} ms")

if __name__ == '__main__':
    model = profile_training()
    profile_inference(model)
