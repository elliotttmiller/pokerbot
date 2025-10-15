"""
DeepStack Poker Neural Network Trainer (Python)
Implements training loop for DeepStack-style neural nets using DataStream and NetBuilder.
"""
import torch
import torch.optim as optim
import torch.nn as nn
from .net_builder import NetBuilder
from .data_stream import DataStream

class MaskedHuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, y_pred, y_true, mask):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(self.delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss * mask)

class DeepStackTrainer:
    def __init__(self, num_buckets, data_path, batch_size=32, hidden_sizes=[128,128], activation='relu', bet_sizing=None, bucket_count=None, use_gpu=False, lr=0.001, epochs=10):
        self.net = NetBuilder.build_net(num_buckets, hidden_sizes, activation, use_gpu)
        self.data_stream = DataStream(data_path, batch_size, use_gpu)
        self.criterion = MaskedHuberLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.epochs = epochs
        self.use_gpu = use_gpu
        self.bet_sizing = bet_sizing
        self.bucket_count = bucket_count
    def train(self):
        self.best_val_loss = float('inf')
        self.best_model_state = None
        patience_limit = 5
        patience_counter = 0
        # Adaptive LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.5)
        # Continual model refresh: load best model if exists
        import os
        best_model_path = "models/pretrained/best_model.pt"
        if os.path.exists(best_model_path):
            self.net.load_state_dict(torch.load(best_model_path))
            print("[INFO] Loaded best model for continual refresh.")
        for epoch in range(self.epochs):
            self.data_stream.start_epoch()
            train_batches = self.data_stream.get_train_batch_count()
            for batch_idx in range(train_batches):
                inputs, targets, mask = self.data_stream.get_batch('train', batch_idx)
                inputs = torch.tensor(inputs)
                targets = torch.tensor(targets)
                mask = torch.tensor(mask)
                if self.use_gpu:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    mask = mask.cuda()
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets, mask)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs} complete.")
            # Validation and model selection
            valid_batches = self.data_stream.get_valid_batch_count()
            valid_loss = 0.0
            for batch_idx in range(valid_batches):
                v_inputs, v_targets, v_mask = self.data_stream.get_batch('valid', batch_idx)
                v_inputs = torch.tensor(v_inputs)
                v_targets = torch.tensor(v_targets)
                v_mask = torch.tensor(v_mask)
                if self.use_gpu:
                    v_inputs = v_inputs.cuda()
                    v_targets = v_targets.cuda()
                    v_mask = v_mask.cuda()
                v_outputs = self.net(v_inputs)
                v_loss = self.criterion(v_outputs, v_targets, v_mask)
                valid_loss += v_loss.item()
            valid_loss /= valid_batches
            print(f"Validation loss after epoch {epoch+1}: {valid_loss:.4f}")
            # Adaptive LR step
            scheduler.step(valid_loss)
            # Early stopping and best model checkpoint
            if valid_loss < self.best_val_loss:
                self.best_val_loss = valid_loss
                self.best_model_state = self.net.state_dict()
                torch.save(self.best_model_state, best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience_limit:
                    print("[INFO] Early stopping triggered.")
                    break
            # Save regular checkpoint
            torch.save(self.net.state_dict(), f"models/pretrained/epoch_{epoch+1}.pt")
