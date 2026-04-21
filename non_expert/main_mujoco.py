import os
import sys
import time

from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np

from core.config import Config, load_yaml
from core.nn_policy import MINTPolicy
from non_expert.helpers import J_upper_bound, Q_upper_bound, Tee, find_fixed_point_v2
from non_expert.mujoco_system import MujocoSystem


def make_dataset(X, U, C, X_next):
    return [(x, u, c, x_next) for x, u, c, x_next in zip(X, U, C, X_next)]


def prune_dataset(X, U, C, X_next, Q, lambd):
    J_ub = J_upper_bound(X, X, Q, lambd)
    to_prune = np.where(Q > J_ub)[0]
    return (
        np.delete(X, to_prune, axis=0),
        np.delete(U, to_prune, axis=0),
        np.delete(C, to_prune, axis=0),
        np.delete(X_next, to_prune, axis=0),
        np.delete(Q, to_prune, axis=0),
        len(to_prune),
    )


def collect_trajectory(system, policy=None, sigma=0.0, seed=None, max_steps=None):
    obs = system.reset(seed=seed)
    X, U, C, X_next = [], [], [], []
    steps = 0

    while True:
        if policy is None:
            u = system.env.action_space.sample()
        else:
            u = policy.make_step(obs.reshape(1, -1)).reshape(-1)
            u = u + np.random.normal(scale=sigma, size=system.nu)
            u = system.clip_action(u)

        cost, obs_next, done = system.transition(u)

        X.append(obs)
        U.append(u)
        C.append(cost)
        X_next.append(obs_next)

        obs = obs_next
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break

    return (
        np.asarray(X, dtype=float),
        np.asarray(U, dtype=float),
        np.asarray(C, dtype=float),
        np.asarray(X_next, dtype=float),
    )


def evaluate_policy(system, policy, config, seed_offset=0):
    returns = []
    for i in range(config.eval_samples):
        obs = system.reset(seed=seed_offset + i)
        G = 0.0
        discount = 1.0

        while True:
            u = policy.make_step(obs.reshape(1, -1)).reshape(-1)
            u = system.clip_action(u)
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
    
    # Name of the environment to be tested.
    env_id = 'InvertedPendulum-v5'
    
    # Load configuration
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, "mujoco_config.yaml")
    config = load_mujoco_config(cfg_path, env_id)

    # Make folder to save results
    hms_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    results_path = os.path.join(path, "results", "mujoco", hms_time)
    os.makedirs(results_path, exist_ok=True)

    with open(os.path.join(results_path, "config.json"), encoding="utf-8", mode="w") as f:
        f.write(config.tojson())

    # Redirect output to output.txt
    output_file = open(os.path.join(results_path, "output.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, output_file)


    env = gym.make(config.env_id)
    system = MujocoSystem(env)

    print(f"Environment: {config.env_id}")
    print(f"Observation dim: {system.nx}, action dim: {system.nu}, full state dim: {system.state_dim}")

    X, U, C, X_next = collect_trajectory(system, policy=None, seed=0)
    D = make_dataset(X, U, C, X_next)
    Q = find_fixed_point_v2(D, gamma=config.gamma, lambd=config.lambd, tol=config.q_tol, max_iter=config.q_iter)
    X, U, C, X_next, Q, pruned = prune_dataset(X, U, C, X_next, Q, config.lambd)
    print(f"initial trajectory samples={len(D)}, pruned={pruned}, dataset size={len(X)}")

    median_values = np.zeros(config.max_iter + 1)
    dataset_sizes = np.zeros(config.max_iter + 1)

    pi_mint = MINTPolicy(nx=system.nx, nu=system.nu, k=config.k, lambd=config.lambd)
    pi_mint.set_data(X, U, Q)
    eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=10_000)
    median_values[0] = np.median(eval_returns)
    dataset_sizes[0] = len(X)
    print(f"iteration=0, median value={median_values[0]:.1f}, dataset size={len(X)}")

    for t in range(1, config.max_iter + 1):
        X_new, U_new, C_new, X_next_new = collect_trajectory(
            system,
            policy=pi_mint,
            sigma=config.sigma,
            seed=t,
        )
        print(f"\niteration={t}, collected {len(X_new)} new samples")

        Q_new = Q_upper_bound(X_new, U_new, X, U, q=Q, lambd=config.lambd)
        TQ_new = C_new + config.gamma * J_upper_bound(X_next_new, X, Q, config.lambd)
        improves = TQ_new < Q_new
        print(f"improving points={np.sum(improves)}/{len(TQ_new)}")

        X = np.concatenate([X, X_new[improves]], axis=0)
        U = np.concatenate([U, U_new[improves]], axis=0)
        C = np.concatenate([C, C_new[improves]], axis=0)
        X_next = np.concatenate([X_next, X_next_new[improves]], axis=0)

        D = make_dataset(X, U, C, X_next)
        Q = find_fixed_point_v2(D, gamma=config.gamma, lambd=config.lambd, tol=config.q_tol, max_iter=config.q_iter, warm_Q=Q)
        X, U, C, X_next, Q, pruned = prune_dataset(X, U, C, X_next, Q, config.lambd)

        pi_mint.set_data(X, U, Q)

        eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=10_000)
        median_values[t] = np.median(eval_returns)
        dataset_sizes[t] = len(X)
        print(f"iteration={t}, pruned={pruned}, median value={median_values[t]:.3f}, dataset size={len(X)}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(np.arange(config.max_iter + 1), median_values)
    axs[0].set_title("Median Return")
    axs[0].set_xlabel("Episode")
    axs[0].set_ylabel("Undiscounted Return")
    axs[0].grid(alpha=0.5)

    axs[1].plot(np.arange(config.max_iter + 1), dataset_sizes)
    axs[1].set_title("Dataset Size")
    axs[1].set_xlabel("Episode")
    axs[1].set_ylabel("Samples")
    axs[1].grid(alpha=0.5)

    plt.suptitle(f"MINT policy on {config.env_id}")

    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"{hms_time}_{config.env_id}_lambda{config.lambd}.pdf"))

    np.savez(
        os.path.join(results_path, "mujoco_results.npz"),
        X=X,
        U=U,
        C=C,
        X_next=X_next,
        Q=Q,
        median_values=median_values,
        dataset_sizes=dataset_sizes,
        env_id=config.env_id,
    )
    env.close()
