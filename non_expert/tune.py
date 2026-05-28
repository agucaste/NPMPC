import argparse
import optuna
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import Config, load_yaml
from non_expert.main import train, load_mujoco_config


def objective(trial, args):
    """Run one Optuna trial by sampling config values and calling training.

    Args:
        trial: Optuna trial object used for suggestions, reporting, and pruning.
        args: Parsed command-line arguments controlling study configuration.

    Returns:
        Scalar training objective returned by ``train``.
    """
    config = load_mujoco_config(args.config_path, args.env_id)

    # policy_type = trial.suggest_categorical("policy_type", ["mint", "chain"])
    policy_type = "chain"
    explore_with_max_bootstrap = config.explore_with_max_bootstrap
    if policy_type == "mint":
        explore_with_max_bootstrap = False

    trial_seed = args.seed * 100_000 + trial.number
    config.recursive_update(
        {
            # Random search over these parameters:
            "lambd": trial.suggest_float("lambd", 1, 100.0, log=True),
            "sigma": trial.suggest_float("sigma", 0.1, 3.0, log=True),
            "td_slack": trial.suggest_float("td_slack", 0.1, 1.0),
            "anneal_sigma": trial.suggest_categorical("anneal_sigma", [False, True]),
            "gamma": trial.suggest_categorical("gamma", [0.99, 0.999]),
            
            "policy_type": policy_type,
            "explore_with_max_bootstrap": explore_with_max_bootstrap,
            "seed": trial_seed,
            "save_plots": False,
            "save_dataset": False,
            "plot_Q_values": False,
            "wandb_name": f"{args.study_name}-trial-{trial.number}-{args.env_id}",
        },
    )

    results_tag = f"trial-{trial.number}-seed-{trial_seed}"
    score = train(config, trial=trial, results_tag=results_tag)
    return float(score)


def make_sampler(name, seed):
    """Create an Optuna sampler by name.

    Args:
        name: Sampler name. Supported values are ``random`` and ``tpe``.
        seed: Random seed passed to deterministic samplers.

    Returns:
        Configured Optuna sampler.
    """
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed, multivariate=True)
    return optuna.samplers.RandomSampler(seed=seed)


def make_pruner(name):
    """Create an Optuna pruner by name.

    Args:
        name: Pruner name. Supported values are ``none`` and ``median``.

    Returns:
        Configured Optuna pruner.
    """
    if name == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=25,
        interval_steps=10,
        n_min_trials=5,
    )


def parse_args():
    """Parse command-line arguments for Optuna tuning."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="Swimmer-v5")
    parser.add_argument("--config-path", type=str, default="non_expert/mujoco_config.yaml")
    parser.add_argument("--study-name", type=str, default="npmpc_optuna")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_npmpc.db")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--sampler", choices=["random", "tpe"], default="random")
    parser.add_argument("--pruner", choices=["none", "median"], default="median")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers-hint", type=int)
    return parser.parse_args()


def resolve_config_path(config_path):
    """Resolve a config path relative to the CWD or project root.

    Args:
        config_path: User-provided config path.

    Returns:
        Existing config path as a string.

    Raises:
        FileNotFoundError: If the config path cannot be resolved.
    """
    path = Path(config_path)
    if path.exists():
        return str(path)

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return str(project_path)

    raise FileNotFoundError(f"Config path does not exist: {config_path}")


def summarize_study(study):
    """Print a compact summary of the completed Optuna study.

    Args:
        study: Optuna study after optimization.
    """
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    pruned = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials)
    failed = sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Completed trials: {completed}")
    print(f"Pruned trials: {pruned}")
    print(f"Failed trials: {failed}")


def main():
    """Create and optimize an Optuna study for non-expert MuJoCo training."""
    args = parse_args()

    random.seed(args.seed)
    args.config_path = resolve_config_path(args.config_path)

    if args.num_workers_hint is not None and args.num_workers_hint > 1:
        print(
            "num-workers-hint is informational only; launch multiple tune.py "
            "processes with the same study-name and storage for parallel trials.",
        )

    # Recommended before launching many workers:
    #   export OMP_NUM_THREADS=1
    #   export MKL_NUM_THREADS=1
    #   export OPENBLAS_NUM_THREADS=1
    #   export NUMEXPR_NUM_THREADS=1
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=make_sampler(args.sampler, args.seed),
        pruner=make_pruner(args.pruner),
    )
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials, n_jobs=1)
    summarize_study(study)


if __name__ == "__main__":
    main()


# Example:
#   python non_expert/tune.py --study-name npmpc_swimmer --storage sqlite:///optuna_npmpc.db
#   python non_expert/tune.py --study-name npmpc_swimmer --storage sqlite:///optuna_npmpc.db
#
# For many workers, prefer PostgreSQL storage:
#   --storage postgresql://USER:PASSWORD@localhost/optuna
