
"""
Automated advanced analysis and reporting for DeepStack agent training.
Generates metrics, loss curves, and strategy visualizations, and exports results to /models/reports.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
pythonpath = os.environ.get("PYTHONPATH")
if pythonpath:
    for p in pythonpath.split(os.pathsep):
        if p and p not in sys.path:
            sys.path.insert(0, p)
# Fallback: always add src path directly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.agents.cfr_agent import CFRAgent

def run_analysis_report(samples_dir='src/train_samples', report_dir='models/reports'):
    os.makedirs(report_dir, exist_ok=True)
    # Load .pt files using torch
    X = torch.load(os.path.join(samples_dir, 'train_inputs.pt'))
    y = torch.load(os.path.join(samples_dir, 'train_targets.pt'))
    X = X.float().numpy()
    y = y.float().numpy()
    # Train/val split
    n = X.shape[0]
    split = max(1, int(n * 0.8))
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    # Simple PyTorch MLP
    class MLP(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            )
        def forward(self, x):
            return self.net(x)
    model = MLP(X.shape[1], y.shape[1] if y.ndim > 1 else 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_losses, val_losses = [], []
    epochs = 10
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        pred = model(x_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # Validation
        model.eval()
        with torch.no_grad():
            x_v = torch.tensor(X_val, dtype=torch.float32)
            y_v = torch.tensor(y_val, dtype=torch.float32)
            val_pred = model(x_v)
            val_loss = loss_fn(val_pred, y_v)
            val_losses.append(val_loss.item())
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")
    # Plot loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
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
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'avg_strategy': avg_strategy.tolist()
    }
    with open(os.path.join(report_dir, 'metrics.json'), 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    print(f"Analysis report generated in {report_dir}")

if __name__ == '__main__':
    run_analysis_report()
