"""
Automated resource monitoring for DeepStack training and inference.
Logs CPU, memory, and (if available) GPU utilization during training and inference.
"""
import psutil
import time
import threading
import numpy as np
import tensorflow as tf
from src.agents.champion_agent import load_deepstack_train_samples, train_value_network_on_deepstack_samples

class ResourceMonitor:
    def __init__(self, interval=1.0):
        self.interval = interval
        self.running = False
        self.cpu = []
        self.mem = []
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()
    def stop(self):
        self.running = False
        self.thread.join()
    def _monitor(self):
        while self.running:
            self.cpu.append(psutil.cpu_percent())
            self.mem.append(psutil.virtual_memory().percent)
            time.sleep(self.interval)
    def report(self):
        print(f"CPU usage: min={min(self.cpu):.1f}%, max={max(self.cpu):.1f}%, avg={np.mean(self.cpu):.1f}%")
        print(f"Memory usage: min={min(self.mem):.1f}%, max={max(self.mem):.1f}%, avg={np.mean(self.mem):.1f}%")

if __name__ == '__main__':
    monitor = ResourceMonitor(interval=0.5)
    samples = load_deepstack_train_samples('data/train_samples')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(13)
    ])
    monitor.start()
    history = train_value_network_on_deepstack_samples(model, samples, epochs=3, batch_size=32)
    monitor.stop()
    monitor.report()
    # Inference monitoring
    monitor = ResourceMonitor(interval=0.2)
    dummy_inputs = np.random.rand(100, 27).astype(np.float32)
    monitor.start()
    _ = model.predict(dummy_inputs, batch_size=32, verbose=0)
    monitor.stop()
    monitor.report()
