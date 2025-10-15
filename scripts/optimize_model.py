import optuna
import mlflow
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from deepstack.core.deepstack_trainer import DeepStackTrainer
from deepstack.core.data_stream import DataStream
import json

# Load config
with open('scripts/config/training.json') as f:
    config = json.load(f)

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs = trial.suggest_int('epochs', 10, 50)
    # Update config
    config['lr'] = lr
    config['batch_size'] = batch_size
    config['epochs'] = epochs
    config['train_batch_size'] = batch_size
    # Data and trainer
    data_path = config.get('data_path', '../data/deepstacked_training/samples/train_samples')
    data_stream = DataStream(data_path, batch_size)
    trainer = DeepStackTrainer(config, data_stream)
    # MLflow tracking
    with mlflow.start_run():
        mlflow.log_params({'lr': lr, 'batch_size': batch_size, 'epochs': epochs})
        trainer.train()
        val_loss = trainer.best_val_loss
        mlflow.log_metric('val_loss', val_loss)
    return val_loss

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
print('Best hyperparameters:', study.best_params)

# Ensemble/meta-learning entry point (placeholder)
def run_ensemble(net_class, epochs):
    # Example: load multiple best models and average predictions
    model_paths = [f"models/pretrained/epoch_{i}.pt" for i in range(1, epochs+1)]
    models = []
    for path in model_paths:
        if os.path.exists(path):
            net = net_class()
            net.load_state_dict(torch.load(path))
            models.append(net)
    # ...ensemble logic here...
    print(f"Loaded {len(models)} models for ensemble/meta-learning.")

if __name__ == "__main__":
    # Run Optuna hyperparameter search and MLflow tracking
    # Optionally run ensemble/meta-learning
    # You must pass the correct net class and epochs after Optuna search
    # Example usage after study.optimize:
    # run_ensemble(DeepStackTrainer(config, data_stream).net.__class__, config['epochs'])
    pass