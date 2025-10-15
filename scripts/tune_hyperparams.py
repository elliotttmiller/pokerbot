#!/usr/bin/env python3
"""CLI for hyperparameter tuning using Optuna.

Usage examples:
  # Baseline DQN tuning
  python scripts/tune_hyperparams.py --target dqn --trials 30 --episodes 400 --batch-size 64

  # Champion agent tuning (ensemble weights)
  python scripts/tune_hyperparams.py --target champion --trials 40 --hands 60
"""
import argparse

from src.evaluation.tuning import ChampionTuningConfig, TuningConfig, run_optuna_study


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--study-name", type=str, default="dqn_tuning")
    p.add_argument("--target", choices=["dqn", "champion"], default="dqn")
    p.add_argument("--hands", type=int, default=30, help="Champion tuning: hands per trial")
    p.add_argument("--cfr-warmup", type=int, default=200, help="Champion tuning: CFR warmup iterations")
    p.add_argument("--no-deepstack", action="store_true", help="Champion tuning: disable DeepStack components")
    args = p.parse_args()

    if args.target == "champion":
        cfg = ChampionTuningConfig(
            hands=args.hands,
            batch_size=args.batch_size,
            cfr_warmup=args.cfr_warmup,
            storage=args.storage,
            study_name=args.study_name,
            use_deepstack=not args.no_deepstack,
        )
    else:
        cfg = TuningConfig(
            episodes=args.episodes,
            batch_size=args.batch_size,
            storage=args.storage,
            study_name=args.study_name,
        )

    study = run_optuna_study(n_trials=args.trials, config=cfg, target=args.target)

    print("Best trial:", study.best_trial.number)
    print("Best value:", study.best_value)
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
