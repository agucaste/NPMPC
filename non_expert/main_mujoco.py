import os
import random
import sys
import time
from pathlib import Path

from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np

from core.config import Config, load_yaml
from core.nn_policy import EncodedMINTPolicy, configure_faiss_threads
from non_expert.helpers import (
    J_upper_bound,
    Q_upper_bound,
    Tee,
    find_fixed_point_v2,
    find_fixed_point_v3,
    seed_all,
)
from non_expert.logger import Logger
from non_expert.mujoco_system import GymSystem, MujocoSystem
from non_expert.observation_encoders import encode_trajectories, load_observation_encoder

EVAL_SEED_OFFSET = 11_426


def make_system(env):
    if hasattr(env.unwrapped, "model") and hasattr(env.unwrapped, "data"):
        return MujocoSystem(env)
    return GymSystem(env)


def make_dataset(X, U, C, X_next, system, do_bootstrap=False, max_bootstrap=1):
    if not do_bootstrap:
        return [(x, u, c, x_next) for x, u, c, x_next in zip(X, U, C, X_next)]

    D = []
    for i in range(len(X)):
        transition = []
        last_obs = X[i]

        for step in range(max_bootstrap):
            j = i + step
            if j < len(X):
                transition.extend([X[j], U[j], C[j]])
                last_obs = X_next[j]
            else:
                transition.extend([last_obs, np.zeros(system.nu), 0.0])

        transition.append(last_obs)
        D.append(tuple(transition))

    return D


def make_dataset_from_trajectories(trajectories, system, do_bootstrap=False, max_bootstrap=1):
    D = []
    for X, U, C, X_next in trajectories:
        D.extend(make_dataset(
            X,
            U,
            C,
            X_next,
            system,
            do_bootstrap=do_bootstrap,
            max_bootstrap=max_bootstrap,
        ))
    return D


def prune_dataset(X, U, C, X_next, D, Q, lambd):
    J_ub = J_upper_bound(X, X, Q, lambd)
    to_prune = np.where(Q > J_ub)[0]
    keep = np.ones(len(X), dtype=bool)
    keep[to_prune] = False
    return (
        np.delete(X, to_prune, axis=0),
        np.delete(U, to_prune, axis=0),
        np.delete(C, to_prune, axis=0),
        np.delete(X_next, to_prune, axis=0),
        [d for d, should_keep in zip(D, keep) if should_keep],
        np.delete(Q, to_prune, axis=0),
        len(to_prune),
    )


def empirical_lipschitz_constant(X, Q):
    X = np.asarray(X, dtype=float)
    Q = np.asarray(Q, dtype=float).reshape(-1)

    if X.shape[0] != Q.shape[0]:
        raise ValueError(f"X and Q must have the same number of rows, got {X.shape[0]} and {Q.shape[0]}")
    if X.shape[0] < 2:
        return 0.0

    X = X.reshape(X.shape[0], -1)
    batch_size = int(os.environ.get("MINT_DISTANCE_BATCH_SIZE", 2048))
    x_norm_sq = np.sum(X * X, axis=1)
    max_ratio = 0.0

    for start in range(0, X.shape[0], batch_size):
        stop = min(start + batch_size, X.shape[0])
        dist_sq = x_norm_sq[start:stop, None] + x_norm_sq[None, :] - 2.0 * (X[start:stop] @ X.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        q_diff = np.abs(Q[start:stop, None] - Q[None, :])

        row_idx = np.arange(start, stop)
        dist_sq[np.arange(stop - start), row_idx] = np.nan

        duplicate_mask = dist_sq == 0.0
        if np.any(duplicate_mask & (q_diff > 0.0)):
            return float("inf")

        valid = dist_sq > 0.0
        if np.any(valid):
            ratios = q_diff[valid] / np.sqrt(dist_sq[valid])
            max_ratio = max(max_ratio, float(np.max(ratios)))

    return max_ratio


def logged_lipschitz_constant(iteration, X, Q, every, previous):
    every = int(every)
    if every <= 0 or iteration % every != 0:
        return previous
    return empirical_lipschitz_constant(X, Q)


def flatten_trajectories(trajectories):
    X, U, C, X_next = zip(*trajectories)
    return (
        np.concatenate(X, axis=0),
        np.concatenate(U, axis=0),
        np.concatenate(C, axis=0),
        np.concatenate(X_next, axis=0),
    )


def collect_trajectories(system, policy=None, sigma=0.0, seed=None, max_steps=None, num_trajectories: int = 1):
    trajectories = []
    for traj_idx in range(num_trajectories):
        traj_seed = None if seed is None else seed + traj_idx
        obs = system.reset(seed=traj_seed)
        X, U, C, X_next = [], [], [], []
        steps = 0

        while True:
            if policy is None:
                u = system.env.action_space.sample()
                # print(f"u is {u}", f"steps is {steps}")
            else:
                u = policy.make_step(obs.reshape(1, -1)).reshape(-1)
                u = u + np.random.normal(scale=sigma, size=system.nu)

            cost, obs_next, done = system.transition(u)

            X.append(obs)
            U.append(u)
            C.append(cost)
            X_next.append(obs_next)

            obs = obs_next
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break

        trajectories.append((
            np.asarray(X, dtype=float),
            np.asarray(U, dtype=float),
            np.asarray(C, dtype=float),
            np.asarray(X_next, dtype=float),
        ))

    return trajectories


def evaluate_policy(system, policy, config, seed_offset=11_426):
    returns = []
    for i in range(config.eval_samples):
        obs = system.reset(seed=seed_offset + i)
        G = 0.0
        discount = 1.0

        while True:
            u = policy.make_step(obs.reshape(1, -1)).reshape(-1)
            cost, obs, done = system.transition(u)
            G += discount * (-cost)
            # discount *= config.gamma

            if done:
                break

        returns.append(G)

    return np.asarray(returns, dtype=float)


def load_mujoco_config(cfg_path, env_id):
    config_yaml = load_yaml(cfg_path)
    config = Config.dict2config(config_yaml["defaults"])

    env_specs = config_yaml.get(env_id)
    if env_specs is not None:
        config.recursive_update(env_specs)

    if not config.env_id:
        config.env_id = env_id

    return config


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambd", type=float)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--env-id", type=str)
    parser.add_argument("--faiss-threads", type=int)
    args = parser.parse_args()


    # Name of the environment to be tested.
    env_id = args.env_id or "InvertedPendulum-v5"

    # Load configuration
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, "mujoco_config.yaml")
    config = load_mujoco_config(cfg_path, env_id)
    # Override with command-line arguments
    overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None
    }
    config.recursive_update(overrides)
    config.recursive_update({"seed": random.randrange(0, 1000)})
    seed_all(config.seed)

    configure_faiss_threads(config.faiss_threads)
    os.environ["MINT_DISTANCE_BATCH_SIZE"] = str(config.distance_batch_size)

    # Make folder to save results
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(
        path,
        "results",
        "mujoco",
        env_id,
        f"seed-{config.seed}-{hms_time}",
    )
    os.makedirs(results_path, exist_ok=True)

    # Redirect output to output.txt
    output_file = open(os.path.join(results_path, "output.txt"), "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(sys.stdout, output_file)

    logger = Logger(
        output_dir=results_path,
        exp_name=f"{config.env_id}",
        use_wandb=getattr(config, "use_wandb", False),
        config=config,
        wandb_name=f"{config.env_id}-seed-{config.seed}-{hms_time}",
    )
    for key in [
        "Train/Iteration",
        "Data/NewSamples",
        "Data/ImprovingPoints",
        "Data/CollectedPoints",
        "Data/DatasetSize",
        "Data/PrunedPoints",
        "Data/LipschitzEmpirical",
        "FixedPoint/Steps",
        "Bootstrap/Average",
        "Bootstrap/Median",
        "Eval/MedianReturn",
        "Time/Total",
        "Time/Update",
        "Time/Evaluate",
    ]:
        if key in {"Bootstrap/Average", "Bootstrap/Median", "Eval/MedianReturn"}:
            logger.register_key(key, display_format=".1f")
        else:
            logger.register_key(key)


    env = gym.make(config.env_id)
    env.action_space.seed(config.seed)
    system = make_system(env)
    project_root = Path(path).parent
    encoder = load_observation_encoder(config, config.env_id, system.nx, project_root)
    encoder_metadata = getattr(encoder, "metadata", {})
    feature_dim = int(encoder.output_dim)
    do_bootstrap = config.do_bootstrap
    max_bootstrap = config.max_bootstrap

    def solve_Q(D, warm_Q=None):
        if do_bootstrap:
            return find_fixed_point_v3(
                D,
                gamma=config.gamma,
                lambd=config.lambd,
                tol=config.q_tol,
                max_iter=config.q_iter,
                warm_Q=warm_Q,
                K=max_bootstrap,
            )
        Q = find_fixed_point_v2(
            D,
            gamma=config.gamma,
            lambd=config.lambd,
            tol=config.q_tol,
            max_iter=config.q_iter,
            warm_Q=warm_Q,
        )
        return Q, 1.0, 1.0, None

    logger.log(f"Environment: {config.env_id}")
    logger.log(f"Seed: {config.seed}")
    logger.log(
        f"Observation dim: {system.nx}, action dim: {system.nu}, full state dim: {system.state_dim}",
    )
    logger.log(
        f"Encoder: {encoder.name}, feature dim: {feature_dim}, "
        f"path: {encoder_metadata.get('path', '<identity>')}",
    )

    experiment_start_time = time.time()

    raw_trajectories = collect_trajectories(
        system,
        policy=None,
        seed=config.seed,
        num_trajectories=config.traj_per_iter,
        max_steps=config.M,
    )
    trajectories = encode_trajectories(raw_trajectories, encoder)
    X, U, C, X_next = flatten_trajectories(trajectories)

    update_start_time = time.time()
    D = make_dataset_from_trajectories(
        trajectories,
        system,
        do_bootstrap=do_bootstrap,
        max_bootstrap=max_bootstrap,
    )
    initial_samples = len(D)
    Q, avg_bootstrap, median_bootstrap, fixed_point_steps = solve_Q(D)
    X, U, C, X_next, D, Q, pruned = prune_dataset(X, U, C, X_next, D, Q, config.lambd)

    median_returns = np.zeros(config.max_iter + 1)
    dataset_sizes = np.zeros(config.max_iter + 1)
    improving_points = np.zeros(config.max_iter + 1)
    pruned_points = np.zeros(config.max_iter + 1)
    avg_bootstrap_steps = np.ones(config.max_iter + 1)
    median_bootstrap_steps = np.ones(config.max_iter + 1)

    pi_mint = EncodedMINTPolicy(encoder=encoder, nx=feature_dim, nu=system.nu, k=config.k, lambd=config.lambd)
    pi_mint.set_data(X, U, Q)
    update_time = time.time() - update_start_time

    evaluate_start_time = time.time()
    eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=config.seed + EVAL_SEED_OFFSET)
    evaluate_time = time.time() - evaluate_start_time
    median_returns[0] = np.median(eval_returns)
    dataset_sizes[0] = len(X)
    pruned_points[0] = pruned
    avg_bootstrap_steps[0] = avg_bootstrap
    median_bootstrap_steps[0] = median_bootstrap
    lipschitz_empirical = logged_lipschitz_constant(
        0,
        X,
        Q,
        config.lipschitz_log_every,
        previous=np.nan,
    )
    logger.store(
        {
            "Train/Iteration": 0,
            "Data/NewSamples": initial_samples,
            "Data/ImprovingPoints": np.nan,
            "Data/CollectedPoints": np.nan,
            "Data/DatasetSize": len(X),
            "Data/PrunedPoints": pruned,
            "Data/LipschitzEmpirical": lipschitz_empirical,
            "FixedPoint/Steps": np.nan if fixed_point_steps is None else fixed_point_steps,
            "Bootstrap/Average": avg_bootstrap,
            "Bootstrap/Median": median_bootstrap,
            "Eval/MedianReturn": median_returns[0],
            "Time/Total": time.time() - experiment_start_time,
            "Time/Update": update_time,
            "Time/Evaluate": evaluate_time,
        },
    )
    logger.dump_tabular()

    for t in range(1, config.max_iter + 1):
        raw_new_trajectories = collect_trajectories(
            system,
            policy=pi_mint,
            sigma=config.sigma,
            seed=config.seed + t * config.traj_per_iter,
            num_trajectories=config.traj_per_iter,
            max_steps=config.M,
        )
        new_trajectories = encode_trajectories(raw_new_trajectories, encoder)
        X_new, U_new, C_new, X_next_new = flatten_trajectories(new_trajectories)

        update_start_time = time.time()
        Q_new = J_upper_bound(X_new, X, Q, config.lambd)
        TQ_new = C_new + config.gamma * J_upper_bound(X_next_new, X, Q, config.lambd)
        improves = TQ_new < Q_new
        improving_points[t] = np.sum(improves)

        pruned = 0
        fixed_point_steps = None
        avg_bootstrap = avg_bootstrap_steps[t - 1]
        median_bootstrap = median_bootstrap_steps[t - 1]
        if np.any(improves):
            D_new = make_dataset_from_trajectories(
                new_trajectories,
                system,
                do_bootstrap=do_bootstrap,
                max_bootstrap=max_bootstrap,
            )

            X = np.concatenate([X, X_new[improves]], axis=0)
            U = np.concatenate([U, U_new[improves]], axis=0)
            C = np.concatenate([C, C_new[improves]], axis=0)
            X_next = np.concatenate([X_next, X_next_new[improves]], axis=0)
            D.extend([d for d, improve in zip(D_new, improves) if improve])
            Q, avg_bootstrap, median_bootstrap, fixed_point_steps = solve_Q(D, warm_Q=Q)
            X, U, C, X_next, D, Q, pruned = prune_dataset(X, U, C, X_next, D, Q, config.lambd)

            pi_mint.set_data(X, U, Q)

        update_time = time.time() - update_start_time

        evaluate_start_time = time.time()
        eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=config.seed + EVAL_SEED_OFFSET)
        evaluate_time = time.time() - evaluate_start_time
        median_returns[t] = np.median(eval_returns)
        dataset_sizes[t] = len(X)
        pruned_points[t] = pruned
        avg_bootstrap_steps[t] = avg_bootstrap
        median_bootstrap_steps[t] = median_bootstrap
        lipschitz_empirical = logged_lipschitz_constant(
            t,
            X,
            Q,
            config.lipschitz_log_every,
            previous=lipschitz_empirical,
        )
        logger.store(
            {
                "Train/Iteration": t,
                "Data/NewSamples": len(X_new),
                "Data/ImprovingPoints": int(improving_points[t]),
                "Data/CollectedPoints": len(TQ_new),
                "Data/DatasetSize": len(X),
                "Data/PrunedPoints": pruned,
                "Data/LipschitzEmpirical": lipschitz_empirical,
                "FixedPoint/Steps": np.nan if fixed_point_steps is None else fixed_point_steps,
                "Bootstrap/Average": avg_bootstrap,
                "Bootstrap/Median": median_bootstrap,
                "Eval/MedianReturn": median_returns[t],
                "Time/Total": time.time() - experiment_start_time,
                "Time/Update": update_time,
                "Time/Evaluate": evaluate_time,
            },
        )
        logger.dump_tabular()

    episodes = np.arange(config.max_iter + 1)
    fig, axs = plt.subplots(1, 4, figsize=(24, 5))
    axs[0].plot(episodes, median_returns)
    axs[0].set_title("Median Return")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Undiscounted Return")
    axs[0].grid(alpha=0.5)

    axs[1].plot(episodes, dataset_sizes)
    axs[1].set_title("Dataset Size")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Samples")
    axs[1].grid(alpha=0.5)

    axs[2].plot(episodes, avg_bootstrap_steps, label="Average")
    axs[2].plot(episodes, median_bootstrap_steps, c='C0', linestyle='--', label="Median")
    axs[2].set_title(f"N-step Bootstrapping statistics (max={max_bootstrap})")
    axs[2].set_xlabel("Episode")
    axs[2].set_ylabel("Bootstrap steps")
    axs[2].grid(alpha=0.5)
    # axs[2].legend(loc="upper left")
    axs[2].legend()
        
    axs[3].scatter(episodes, improving_points, marker='s', label="Improving points", color="C2")
    # ax_counts.plot(episodes, pruned_points, "--", label="Pruned points", color="C3")
    axs[3].set_xlabel("Episode")
    axs[3].set_ylabel("Points")
    axs[3].legend(loc="upper right")
    axs[3].set_title("Improving points across iterations")
    axs[3].grid(alpha=0.5)

    plt.suptitle(f"MINT policy on {config.env_id}")

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{hms_time}_{config.env_id}_lambda{config.lambd}_sigma{config.sigma}.pdf"))

    np.savez(
        os.path.join(results_path, "mujoco_results.npz"),
        X=X,
        U=U,
        C=C,
        X_next=X_next,
        Q=Q,
        median_values=median_returns,
        dataset_sizes=dataset_sizes,
        improving_points=improving_points,
        pruned_points=pruned_points,
        avg_bootstrap_steps=avg_bootstrap_steps,
        median_bootstrap_steps=median_bootstrap_steps,
        env_id=config.env_id,
        encoder_name=encoder.name,
        encoder_path=encoder_metadata.get("path", ""),
        encoder_sha256=encoder_metadata.get("sha256", ""),
        encoder_feature_dim=feature_dim,
    )
    env.close()
    logger.close()
    sys.stdout = original_stdout
    output_file.close()
