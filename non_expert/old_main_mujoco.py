import os
import random
import sys
import time
from pathlib import Path

from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np

from core.config import Config, load_yaml
from core.nn_policy import ChainPolicy, EncodedMINTPolicy, configure_faiss_threads
from non_expert.helpers import (
    J_upper_bound,
    Tee,
    find_fixed_point_v2,
    find_fixed_point_v3,
    find_fixed_point_v4,
    plot_Q_values,
    seed_all,
)
from non_expert.logger import Logger
from non_expert.mujoco_system import GymSystem, MujocoSystem
from non_expert.observation_encoders import encode_trajectories, load_observation_encoder
from non_expert.envs_in_parallel import make_vine_envs, reset_vine_envs_same_state

EVAL_SEED_OFFSET = 1_14_26


def make_system(env):
    if hasattr(env.unwrapped, "model") and hasattr(env.unwrapped, "data"):
        return MujocoSystem(env)
    return GymSystem(env)


def make_dataset(Z, U, C, Z_next, system, do_bootstrap=False, max_bootstrap=1):
    if not do_bootstrap:
        return [(z, u, c, z_next) for z, u, c, z_next in zip(Z, U, C, Z_next)]

    D = []
    for i in range(len(Z)):
        transition = []
        last_feature = Z[i]

        for step in range(max_bootstrap):
            j = i + step
            if j < len(Z):
                transition.extend([Z[j], U[j], C[j]])
                last_feature = Z_next[j]
            else:
                transition.extend([last_feature, np.zeros(system.nu), 0.0])

        transition.append(last_feature)
        D.append(tuple(transition))

    return D


def make_dataset_from_trajectories(trajectories, system, do_bootstrap=False, max_bootstrap=1):
    D = []
    for Z, U, C, Z_next in trajectories:
        D.extend(make_dataset(
            Z,
            U,
            C,
            Z_next,
            system,
            do_bootstrap=do_bootstrap,
            max_bootstrap=max_bootstrap,
        ))
    return D


def optimistic_bootstrap_targets(D, policy, gamma, max_bootstrap, batch_size):
    """Compute optimistic k-step targets for bootstrapped transition rows.

    Args:
        D: Bootstrapped transition rows with ``max_bootstrap`` action-cost steps.
        policy: Fitted MINT-style policy used to evaluate tail upper bounds.
        gamma: Discount factor.
        max_bootstrap: Maximum bootstrap horizon stored in each transition row.
        batch_size: Batch size for nearest-neighbor upper-bound queries.

    Returns:
        The minimum optimistic target over all stored bootstrap horizons for each row.
    """
    assert max_bootstrap >= 1, "max_bootstrap must be at least one."
    assert all(len(d) == 3 * max_bootstrap + 1 for d in D), (
        "Each transition tuple must contain max_bootstrap action-cost steps plus a final feature."
    )

    targets = np.empty((max_bootstrap, len(D)), dtype=float)
    discounted_costs = np.zeros(len(D), dtype=float)

    for step in range(max_bootstrap):
        costs = np.asarray([d[2 + 3 * step] for d in D], dtype=float).reshape(-1)
        z_next = np.asarray([d[3 + 3 * step] for d in D], dtype=float)
        discounted_costs = discounted_costs + (gamma ** step) * costs
        targets[step] = discounted_costs + (gamma ** (step + 1)) * policy.J_upper_bound_knn(
            z_next,
            batch_size=batch_size,
        )

    return np.min(targets, axis=0)


def prune_dataset(Z, U, C, Z_next, D, Q, J_ub, lambd):
    """Prune rows whose stored value is above the current upper bound."""
    to_prune = np.where(Q > J_ub)[0]
    keep = np.ones(len(Z), dtype=bool)
    keep[to_prune] = False
    return (
        np.delete(Z, to_prune, axis=0),
        np.delete(U, to_prune, axis=0),
        np.delete(C, to_prune, axis=0),
        np.delete(Z_next, to_prune, axis=0),
        [d for d, should_keep in zip(D, keep) if should_keep],
        np.delete(Q, to_prune, axis=0),
        len(to_prune),
        keep,
    )


def make_action_chains(D, last_bootstraps, nu):
    """Build pre-trimmed action chains from bootstrapped dataset rows."""
    last_bootstraps = np.asarray(last_bootstraps, dtype=int).reshape(-1)
    assert len(D) == last_bootstraps.shape[0], "D and last_bootstraps must have the same length"

    action_chains = []
    for d, bootstrap in zip(D, last_bootstraps):
        assert bootstrap >= 1, "Bootstrap horizons must be at least one."
        assert bootstrap <= (len(d) - 1) // 3, "Bootstrap horizon exceeds stored actions."
        actions = [
            np.asarray(d[1 + 3 * step], dtype=float).reshape(nu)
            for step in range(bootstrap)
        ]
        action_chains.append(np.asarray(actions, dtype=float).reshape(-1, nu))

    return action_chains


def empirical_lipschitz_constant(Z, Q, batch_size):
    Z = np.asarray(Z, dtype=float)
    Q = np.asarray(Q, dtype=float).reshape(-1)

    if Z.shape[0] != Q.shape[0]:
        raise ValueError(f"Z and Q must have the same number of rows, got {Z.shape[0]} and {Q.shape[0]}")
    if Z.shape[0] < 2:
        return 0.0

    Z = Z.reshape(Z.shape[0], -1)
    batch_size = max(1, int(batch_size))
    z_norm_sq = np.sum(Z * Z, axis=1)
    max_ratio = 0.0

    for start in range(0, Z.shape[0], batch_size):
        stop = min(start + batch_size, Z.shape[0])
        dist_sq = z_norm_sq[start:stop, None] + z_norm_sq[None, :] - 2.0 * (Z[start:stop] @ Z.T)
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


def logged_lipschitz_constant(iteration, Z, Q, every, previous, batch_size):
    every = int(every)
    if every <= 0 or iteration % every != 0:
        return previous
    return empirical_lipschitz_constant(Z, Q, batch_size=batch_size)


def flatten_trajectories(trajectories):
    Z, U, C, Z_next = zip(*trajectories)
    return (
        np.concatenate(Z, axis=0),
        np.concatenate(U, axis=0),
        np.concatenate(C, axis=0),
        np.concatenate(Z_next, axis=0),
    )


def assert_feature_dim(Z, Z_next, feature_dim):
    assert Z.shape[1] == feature_dim, f"Expected feature dim {feature_dim}, got Z shape {Z.shape}"
    assert Z_next.shape[1] == feature_dim, f"Expected feature dim {feature_dim}, got Z_next shape {Z_next.shape}"


def collect_trajectories(system, policy=None, sigma=0.0, seed=None, max_steps=None, num_trajectories: int = 1):
    trajectories = []
    for traj_idx in range(num_trajectories):
        traj_seed = None if seed is None else seed + traj_idx
        obs = system.reset(seed=traj_seed)
        if hasattr(policy, "reset_chain"):
            policy.reset_chain()
        obs_list, U, C, obs_next_list = [], [], [], []
        steps = 0

        while True:
            if policy is None:
                u = system.env.action_space.sample()
                # print(f"u is {u}", f"steps is {steps}")
            else:
                u = policy.make_step(obs.reshape(1, -1)).reshape(-1)
                u = u + np.random.normal(scale=sigma, size=system.nu)

            cost, obs_next, done = system.transition(u)

            obs_list.append(obs)
            U.append(u)
            C.append(cost)
            obs_next_list.append(obs_next)

            obs = obs_next
            steps += 1
            if done or (max_steps is not None and steps >= max_steps):
                break

        trajectories.append((
            np.asarray(obs_list, dtype=float),
            np.asarray(U, dtype=float),
            np.asarray(C, dtype=float),
            np.asarray(obs_next_list, dtype=float),
        ))

    return trajectories


def collect_trajectories_vine(
    vine_envs,
    policy,
    encoder,
    config,
    sigma=0.0,
    seed=None,
    max_steps=None,
    num_trajectories: int = 1,
):
    """Collect trajectories by branching noisy actions in synchronized MuJoCo envs."""
    if policy is None:
        raise ValueError("collect_trajectories_vine requires a fitted policy.")

    trajectories = []
    action_low = vine_envs.single_action_space.low
    action_high = vine_envs.single_action_space.high
    action_dim = action_low.shape[0]
    rng = np.random.default_rng(seed)

    for traj_idx in range(num_trajectories):
        traj_seed = None if seed is None else seed + traj_idx
        obs_all, _ = reset_vine_envs_same_state(vine_envs, seed=traj_seed)
        obs = obs_all[0]
        if hasattr(policy, "reset_chain"):
            policy.reset_chain()

        obs_list, U, C, obs_next_list = [], [], [], []
        steps = 0

        while True:
            u_base = policy.make_step(obs.reshape(1, -1), one_action=True).reshape(-1)
            noise = rng.normal(scale=sigma, size=(vine_envs.num_envs, action_dim))
            actions = np.clip(u_base[None, :] + noise, action_low, action_high)

            # Computation of kappa-step bootstrap
            obs_next_all, rewards, terminated, _, _ = vine_envs.step(actions)
            costs = -np.asarray(rewards, dtype=float)
            z_next_all = encoder.encode(obs_next_all)
            TQ = costs + config.gamma * policy.J_upper_bound_knn(
                z_next_all,
                batch_size=config.distance_batch_size,
            )
            # Optimistic bootstrap right here.
            best_idx = int(np.argmin(TQ))

            obs_list.append(obs)
            U.append(actions[best_idx])
            C.append(costs[best_idx])
            obs_next_list.append(obs_next_all[best_idx])

            steps += 1
            if terminated[best_idx] or (max_steps is not None and steps >= max_steps):
                break

            states = vine_envs.call("get_mujoco_state")
            obs_all = np.stack(vine_envs.call("set_mujoco_state", states[best_idx]))
            obs = obs_all[0]

        trajectories.append((
            np.asarray(obs_list, dtype=float),
            np.asarray(U, dtype=float),
            np.asarray(C, dtype=float),
            np.asarray(obs_next_list, dtype=float),
        ))

    return trajectories


def collect_trajectories_vine_chain(
    vine_envs,
    policy,
    encoder,
    config,
    sigma=0.0,
    seed=None,
    max_steps=None,
    num_trajectories: int = 1,
):
    """Collect vine trajectories by branching and scoring noisy ChainPolicy prefixes."""
    if not hasattr(policy, "make_chain"):
        raise TypeError("collect_trajectories_vine_chain requires a policy with make_chain.")

    trajectories = []
    action_low = vine_envs.single_action_space.low
    action_high = vine_envs.single_action_space.high
    action_dim = action_low.shape[0]
    rng = np.random.default_rng(seed)

    for traj_idx in range(num_trajectories):
        traj_seed = None if seed is None else seed + traj_idx
        obs_all, _ = reset_vine_envs_same_state(vine_envs, seed=traj_seed)
        obs = obs_all[0]
        if hasattr(policy, "reset_chain"):
            policy.reset_chain()

        obs_list, U, C, obs_next_list = [], [], [], []
        steps = 0

        while max_steps is None or steps < max_steps:
            base_chain = np.asarray(policy.make_chain(obs.reshape(1, -1)), dtype=float).reshape(-1, action_dim)
            remaining_steps = base_chain.shape[0] if max_steps is None else min(base_chain.shape[0], max_steps - steps)
            if remaining_steps <= 0:
                break

            base_chain = base_chain[:remaining_steps]
            noise = rng.normal(scale=sigma, size=(vine_envs.num_envs, remaining_steps, action_dim))
            chains = np.clip(base_chain[None, :, :] + noise, action_low, action_high)

            obs_before_steps = []
            obs_after_steps = []
            states_after_steps = []
            costs = np.empty((remaining_steps, vine_envs.num_envs), dtype=float)
            terminated_steps = np.zeros((remaining_steps, vine_envs.num_envs), dtype=bool)

            for chain_step in range(remaining_steps):
                obs_before_steps.append(obs_all.copy())
                obs_next_all, rewards, terminated, _, _ = vine_envs.step(chains[:, chain_step, :])
                costs[chain_step] = -np.asarray(rewards, dtype=float)
                terminated_steps[chain_step] = np.asarray(terminated, dtype=bool)
                obs_after_steps.append(obs_next_all.copy())
                states_after_steps.append(vine_envs.call("get_mujoco_state"))
                obs_all = obs_next_all

                has_terminated = np.any(terminated_steps[:chain_step + 1], axis=0)
                if np.all(has_terminated):
                    break

            rolled_steps = len(obs_after_steps)
            prefix_scores = np.full((vine_envs.num_envs, rolled_steps), np.inf, dtype=float)
            discounted_costs = np.zeros(vine_envs.num_envs, dtype=float)
            valid_prefix = np.ones(vine_envs.num_envs, dtype=bool)

            for chain_step in range(rolled_steps):
                discounted_costs[valid_prefix] += (config.gamma ** chain_step) * costs[chain_step, valid_prefix]
                terminated = terminated_steps[chain_step]
                terminal_prefix = valid_prefix & terminated
                nonterminal_prefix = valid_prefix & ~terminated

                prefix_scores[terminal_prefix, chain_step] = discounted_costs[terminal_prefix]
                if np.any(nonterminal_prefix):
                    z_next = encoder.encode(obs_after_steps[chain_step][nonterminal_prefix])
                    tail_values = policy.J_upper_bound_knn(
                        z_next,
                        batch_size=config.distance_batch_size,
                    )
                    prefix_scores[nonterminal_prefix, chain_step] = (
                        discounted_costs[nonterminal_prefix]
                        + (config.gamma ** (chain_step + 1)) * tail_values
                    )

                valid_prefix = nonterminal_prefix

            best_idx_flat = int(np.argmin(prefix_scores))
            best_env_idx, best_step_idx = np.unravel_index(best_idx_flat, prefix_scores.shape)
            selected_length = best_step_idx + 1

            for chain_step in range(selected_length):
                obs_list.append(obs_before_steps[chain_step][best_env_idx])
                U.append(chains[best_env_idx, chain_step])
                C.append(costs[chain_step, best_env_idx])
                obs_next_list.append(obs_after_steps[chain_step][best_env_idx])

            steps += selected_length
            if terminated_steps[best_step_idx, best_env_idx]:
                break

            selected_state = states_after_steps[best_step_idx][best_env_idx]
            obs_all = np.stack(vine_envs.call("set_mujoco_state", selected_state))
            obs = obs_all[0]

        trajectories.append((
            np.asarray(obs_list, dtype=float),
            np.asarray(U, dtype=float),
            np.asarray(C, dtype=float),
            np.asarray(obs_next_list, dtype=float),
        ))

    return trajectories


def evaluate_policy(system, policy, config, seed_offset=11_426):
    returns = []
    for i in range(config.eval_samples):
        obs = system.reset(seed=seed_offset + i)
        if hasattr(policy, "reset_chain"):
            policy.reset_chain()
        G = 0.0
        discount = 1.0

        while True:
            # Use either the open-loop chains (ChainPolicy), or closed-loop one step (MINT). Passed through one_action
            u = policy.make_step(obs.reshape(1, -1), one_action=False).reshape(-1)
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
    parser.add_argument("--env-id", type=str, default='Swimmer-v5')
    parser.add_argument("--faiss-threads", type=int)
    parser.add_argument("--policy-type", choices=["mint", "chain"])
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--use-vine-collection", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()


    # Name of the environment to be tested.
    # env_id = args.env_id or "InvertedPendulum-v5"
    env_id = args.env_id

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
        "Data/TDGap",
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
        if key in {"Data/TDGap", "Bootstrap/Average", "Bootstrap/Median", "Eval/MedianReturn"}:
            logger.register_key(key, display_format=".1f")
        else:
            logger.register_key(key)


    env = gym.make(config.env_id)
    env.action_space.seed(config.seed)
    system = make_system(env)
    vine_envs = None
    if config.use_vine_collection:
        if not isinstance(system, MujocoSystem):
            raise ValueError("Vine collection requires a MuJoCo environment with settable state.")
        vine_envs = make_vine_envs(config.env_id, config.num_envs, seed=config.seed)

    project_root = Path(path).parent
    encoder = load_observation_encoder(config, config.env_id, system.nx, project_root)
    encoder_metadata = getattr(encoder, "metadata", {})
    feature_dim = int(encoder.output_dim)
    do_bootstrap = config.do_bootstrap
    max_bootstrap = config.max_bootstrap
    policy_cls = ChainPolicy if config.policy_type == "chain" else EncodedMINTPolicy
    pi_mint = policy_cls(encoder=encoder, nx=feature_dim, nu=system.nu, k=config.k, lambd=config.lambd)

    def solve_Q(D, warm_Q=None):
        """Solve for Q values and normalize bootstrap metadata."""
        if do_bootstrap:
            return find_fixed_point_v4(
                D,
                gamma=config.gamma,
                lambd=config.lambd,
                tol=config.q_tol,
                max_iter=config.q_iter,
                warm_Q=warm_Q,
                K=max_bootstrap,
                neighbors=config.k,
            )
        Q = find_fixed_point_v2(
            D,
            gamma=config.gamma,
            lambd=config.lambd,
            tol=config.q_tol,
            max_iter=config.q_iter,
            warm_Q=warm_Q,
            batch_size=config.distance_batch_size,
        )
        last_bootstraps = np.ones(len(D), dtype=int)
        bootstrap_stats = {"average": 1.0, "median": 1.0}
        return Q, last_bootstraps, bootstrap_stats, None

    def set_policy_data(policy, Z, U, Q, D, last_bootstraps):
        """Set policy data with action chains when using ChainPolicy."""
        if isinstance(policy, ChainPolicy):
            action_chains = make_action_chains(D, last_bootstraps, system.nu)
            policy.set_data(Z, U, Q, action_chains)
        else:
            policy.set_data(Z, U, Q)

    logger.log(f"Environment: {config.env_id}")
    logger.log(f"Seed: {config.seed}")
    logger.log(
        f"Observation dim: {system.nx}, action dim: {system.nu}, full state dim: {system.state_dim}",
    )
    logger.log(
        f"Encoder: {encoder.name}, feature dim: {feature_dim}, "
        f"path: {encoder_metadata.get('path', '<identity>')}",
    )
    logger.log("MINT metric space: encoded observations")
    logger.log(f"Policy type: {config.policy_type}")
    logger.log(
        f"Vine collection: {config.use_vine_collection}, "
        f"num_envs: {config.num_envs if config.use_vine_collection else 0}",
    )

    experiment_start_time = time.time()

    raw_trajectories = collect_trajectories(
        system,
        policy=None,
        seed=config.seed,
        num_trajectories=config.traj_per_iter,
        max_steps=config.M,
    )
    encoded_trajectories = encode_trajectories(raw_trajectories, encoder)
    Z, U, C, Z_next = flatten_trajectories(encoded_trajectories)
    assert_feature_dim(Z, Z_next, feature_dim)

    update_start_time = time.time()
    D = make_dataset_from_trajectories(
        encoded_trajectories,
        system,
        do_bootstrap=do_bootstrap,
        max_bootstrap=max_bootstrap,
    )
    initial_samples = len(D)
    Q, last_bootstraps, bootstrap_stats, fixed_point_steps = solve_Q(D)
    set_policy_data(pi_mint, Z, U, Q, D, last_bootstraps)
    # distances = pi_mint.distances_to_dataset(Z, batch_size=config.distance_batch_size)
    J_ub = pi_mint.J_upper_bound_knn(Z, batch_size=config.distance_batch_size)
    Z, U, C, Z_next, D, Q, pruned, keep = prune_dataset(
        Z,
        U,
        C,
        Z_next,
        D,
        Q,
        J_ub,
        config.lambd,
    )
    last_bootstraps = last_bootstraps[keep]

    median_returns = np.zeros(config.episodes + 1)
    dataset_sizes = np.zeros(config.episodes + 1)
    improving_points = np.zeros(config.episodes + 1)
    pruned_points = np.zeros(config.episodes + 1)
    avg_bootstrap_steps = np.ones(config.episodes + 1)
    median_bootstrap_steps = np.ones(config.episodes + 1)

    set_policy_data(pi_mint, Z, U, Q, D, last_bootstraps)

    plot_Q_values(pi_mint,results_path, 0)

    assert pi_mint.nx == feature_dim, f"Expected MINT feature dim {feature_dim}, got {pi_mint.nx}"
    update_time = time.time() - update_start_time

    evaluate_start_time = time.time()
    eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=config.seed + EVAL_SEED_OFFSET)
    evaluate_time = time.time() - evaluate_start_time
    median_returns[0] = np.median(eval_returns)
    dataset_sizes[0] = len(Z)
    pruned_points[0] = pruned
    avg_bootstrap_steps[0] = bootstrap_stats["average"]
    median_bootstrap_steps[0] = bootstrap_stats["median"]
    lipschitz_empirical = logged_lipschitz_constant(
        0,
        Z,
        Q,
        config.lipschitz_log_every,
        previous=np.nan,
        batch_size=config.distance_batch_size,
    )
    logger.store(
        {
            "Train/Iteration": 0,
            "Data/NewSamples": initial_samples,
            "Data/ImprovingPoints": np.nan,
            "Data/TDGap": np.nan,
            "Data/DatasetSize": len(Z),
            "Data/PrunedPoints": pruned,
            "Data/LipschitzEmpirical": lipschitz_empirical,
            "FixedPoint/Steps": np.nan if fixed_point_steps is None else fixed_point_steps,
            "Bootstrap/Average": bootstrap_stats["average"],
            "Bootstrap/Median": bootstrap_stats["median"],
            "Eval/MedianReturn": median_returns[0],
            "Time/Total": time.time() - experiment_start_time,
            "Time/Update": update_time,
            "Time/Evaluate": evaluate_time,
        },
    )
    logger.dump_tabular()

    # Initialize sigma here (and anneal it if provided)
    sigma = config.sigma

    for t in range(1, config.episodes + 1):
        if config.use_vine_collection and isinstance(pi_mint, ChainPolicy):
            raw_new_trajectories = collect_trajectories_vine_chain(
                vine_envs,
                policy=pi_mint,
                encoder=encoder,
                config=config,
                sigma=sigma,
                seed=config.seed + t * config.traj_per_iter,
                num_trajectories=config.traj_per_iter,
                max_steps=config.M,
            )
        elif config.use_vine_collection:
            raw_new_trajectories = collect_trajectories_vine(
                vine_envs,
                policy=pi_mint,
                encoder=encoder,
                config=config,
                sigma=sigma,
                seed=config.seed + t * config.traj_per_iter,
                num_trajectories=config.traj_per_iter,
                max_steps=config.M,
            )
        else:
            raw_new_trajectories = collect_trajectories(
                system,
                policy=pi_mint,
                sigma=sigma,
                seed=config.seed + t * config.traj_per_iter,
                num_trajectories=config.traj_per_iter,
                max_steps=config.M,
            )
        encoded_new_trajectories = encode_trajectories(raw_new_trajectories, encoder)
        Z_new, U_new, C_new, Z_next_new = flatten_trajectories(encoded_new_trajectories)
        assert_feature_dim(Z_new, Z_next_new, feature_dim)

        update_start_time = time.time()
        # distances_new = pi_mint.distances_to_dataset(Z_new, batch_size=config.distance_batch_size)
        # distances_next_new = pi_mint.distances_to_dataset(Z_next_new, batch_size=config.distance_batch_size)
        Q_new = pi_mint.J_upper_bound_knn(Z_new, batch_size=config.distance_batch_size)
        D_new = None
        if do_bootstrap:
            D_new = make_dataset_from_trajectories(
                encoded_new_trajectories,
                system,
                do_bootstrap=do_bootstrap,
                max_bootstrap=max_bootstrap,
            )
            TQ_new = optimistic_bootstrap_targets(
                D_new,
                pi_mint,
                gamma=config.gamma,
                max_bootstrap=max_bootstrap,
                batch_size=config.distance_batch_size,
            )
        else:
            TQ_new = C_new + config.gamma * pi_mint.J_upper_bound_knn(
                Z_next_new,
                batch_size=config.distance_batch_size,
            )
        improves = TQ_new + config.td_slack < Q_new
        improving_points[t] = np.sum(improves)

        gaps = Q_new - TQ_new
        gaps_improving = gaps[improves]
        td_gap_improving = gaps_improving.mean() if gaps_improving.size > 0 else 0
        if gaps_improving.size > 0:
            print(
                "Gaps over improving points: "
                f"min {gaps_improving.min():.1f}, "
                f"median {np.median(gaps_improving):.1f}, "
                f"mean {td_gap_improving:.1f}, "
                f"max {gaps_improving.max():.1f}",
            )
        else:
            print("Gaps over improving points: n/a")

        pruned = 0
        fixed_point_steps = None
        bootstrap_stats = {
            "average": avg_bootstrap_steps[t - 1],
            "median": median_bootstrap_steps[t - 1],
        }
        if np.any(improves):
            if D_new is None:
                D_new = make_dataset_from_trajectories(
                    encoded_new_trajectories,
                    system,
                    do_bootstrap=do_bootstrap,
                    max_bootstrap=max_bootstrap,
                )

            Z = np.concatenate([Z, Z_new[improves]], axis=0)
            U = np.concatenate([U, U_new[improves]], axis=0)
            C = np.concatenate([C, C_new[improves]], axis=0)
            Z_next = np.concatenate([Z_next, Z_next_new[improves]], axis=0)
            D.extend([d for d, improve in zip(D_new, improves) if improve])
            Q, last_bootstraps, bootstrap_stats, fixed_point_steps = solve_Q(D, warm_Q=Q)
            set_policy_data(pi_mint, Z, U, Q, D, last_bootstraps)
            # distances = pi_mint.distances_to_dataset(Z, batch_size=config.distance_batch_size)
            J_ub = pi_mint.J_upper_bound_knn(Z, batch_size=config.distance_batch_size)
            Z, U, C, Z_next, D, Q, pruned, keep = prune_dataset(
                Z,
                U,
                C,
                Z_next,
                D,
                Q,
                J_ub,
                config.lambd,
            )
            last_bootstraps = last_bootstraps[keep]

            set_policy_data(pi_mint, Z, U, Q, D, last_bootstraps)

        if t % 50 == 0:
            plot_Q_values(pi_mint,results_path, t)

        update_time = time.time() - update_start_time

        evaluate_start_time = time.time()
        eval_returns = evaluate_policy(system, pi_mint, config, seed_offset=config.seed + EVAL_SEED_OFFSET)
        evaluate_time = time.time() - evaluate_start_time
        median_returns[t] = np.median(eval_returns)
        dataset_sizes[t] = len(Z)
        pruned_points[t] = pruned
        avg_bootstrap_steps[t] = bootstrap_stats["average"]
        median_bootstrap_steps[t] = bootstrap_stats["median"]
        lipschitz_empirical = logged_lipschitz_constant(
            t,
            Z,
            Q,
            config.lipschitz_log_every,
            previous=lipschitz_empirical,
            batch_size=config.distance_batch_size,
        )
        logger.store(
            {
                "Train/Iteration": t,
                "Data/NewSamples": len(Z_new),
                "Data/ImprovingPoints": int(improving_points[t]),
                "Data/TDGap": td_gap_improving,
                "Data/DatasetSize": len(Z),
                "Data/PrunedPoints": pruned,
                "Data/LipschitzEmpirical": lipschitz_empirical,
                "FixedPoint/Steps": np.nan if fixed_point_steps is None else fixed_point_steps,
                "Bootstrap/Average": bootstrap_stats["average"],
                "Bootstrap/Median": bootstrap_stats["median"],
                "Eval/MedianReturn": median_returns[t],
                "Time/Total": time.time() - experiment_start_time,
                "Time/Update": update_time,
                "Time/Evaluate": evaluate_time,
            },
        )
        logger.dump_tabular()

        if config.anneal_sigma:
            sigma = sigma * (config.episodes - t) / (1 + config.episodes - t)  # Sigma_{t+1} = f(Sigma_t, t)

    episodes = np.arange(config.episodes + 1)
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
        Z=Z,
        U=U,
        C=C,
        Z_next=Z_next,
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
    if vine_envs is not None:
        vine_envs.close()
    env.close()
    logger.close()
    sys.stdout = original_stdout
    output_file.close()
