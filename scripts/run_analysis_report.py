"""
Automated advanced analysis and reporting for DeepStack agent training.
Generates metrics, loss curves, and strategy visualizations, and exports results to /models/reports.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from src.agents.cfr_agent import CFRAgent

def run_analysis_report(samples_dir='data/train_samples', report_dir='models/reports'):
    os.makedirs(report_dir, exist_ok=True)
    samples = load_deepstack_train_samples(samples_dir)
    import tensorflow as tf
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(13)
    ])
    history = train_value_network_on_deepstack_samples(model, samples, epochs=10, batch_size=32)
    # Plot loss curves
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Value Network Training Loss')
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'value_network_loss.png'))
    plt.close()
    # CFR strategy visualization
    agent = CFRAgent()
    infoset = agent.get_infoset('report|infoset', ['fold', 'call', 'raise'])
    infoset.update_regret(0, 1.0)
    infoset.update_regret(1, 2.0)
    infoset.update_regret(2, 3.0)
    avg_strategy = infoset.get_average_strategy()
    plt.figure()
    plt.bar(['fold', 'call', 'raise'], avg_strategy)
    plt.title('CFR Strategy Visualization')
    plt.xlabel('Actions')
    plt.ylabel('Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'cfr_strategy.png'))
    plt.close()
    # Export metrics
    metrics = {
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'avg_strategy': avg_strategy.tolist()
    }
    with open(os.path.join(report_dir, 'metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"Analysis report generated in {report_dir}")

if __name__ == '__main__':
    run_analysis_report()
