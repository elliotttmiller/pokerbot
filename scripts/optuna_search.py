import optuna
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from deepstack.core.train_deepstack import DeepStackTrainer

def objective(trial):
    # Suggest hyperparameters
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    hidden_sizes = [trial.suggest_int('h1', 64, 256), trial.suggest_int('h2', 64, 256)]
    bucket_count = trial.suggest_int('bucket_count', 6, 20)
    bet_sizing = trial.suggest_categorical('bet_sizing', [[1], [1,2], [1,2,3]])
    activation = trial.suggest_categorical('activation', ['relu', 'prelu'])
    epochs = trial.suggest_int('epochs', 5, 20)

    # Update config
    config = {
        "num_buckets": bucket_count,
        "bucket_count": bucket_count,
        "bet_sizing": bet_sizing,
        "hidden_sizes": hidden_sizes,
        "activation": activation,
        "data_path": "../src/train_samples",
        "batch_size": batch_size,
        "use_gpu": True,
        "lr": lr,
    "epochs": epochs,
    "versions_dir": "models/versions",
    "checkpoint_dir": "models/checkpoints"
    }
    with open('scripts/config/training.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Run training
    trainer = DeepStackTrainer(
        num_buckets=config['num_buckets'],
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        hidden_sizes=config['hidden_sizes'],
        activation=config['activation'],
        bet_sizing=config['bet_sizing'],
        bucket_count=config['bucket_count'],
        use_gpu=config['use_gpu'],
        lr=config['lr'],
        epochs=config['epochs']
    )
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        trainer.train()
        # Assume trainer stores last validation loss
        val_loss = trainer.last_val_loss if hasattr(trainer, 'last_val_loss') else 0.0
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    print("Best trial:", study.best_trial)
