"""
Minimal Minari encoder pretraining example.

This is a 1-step encoder that jointly learns:
    - a latent representation phi(s)
    - a latent dynamics model g(z, a) that predicts the next latent state change
    - a reward predictor h(z, a)

Learns an encoder phi(s) so that:
    z_t = phi(s_t)
    z_t + g(z_t, a_t) ~= phi(s_{t+1})
    h(z_t, a_t) ~= r_t

After training, discard g and h. Keep phi frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from non_expert.minari.utils import (
    DATASET,
    DynamicsEncoder,
    FrozenEncoder,
    compute_batch_losses,
    compute_k_step_batch_losses,
    encode_normalized_observations,
    evaluate_encoder,
    evaluate_loader_losses,
    flatten_k_step_transitions,
    flatten_transitions,
    load_minari_datasets,
    make_k_step_transition_loader,
    make_transition_loader,
    neighborhood_preservation_overlaps,
    plot_latent_scatter,
    plot_latent_spectrum,
    plot_encoder_neighborhood_comparison,
    plot_neighborhood_preservation,
    plot_training_history,
    save_encoder,
)


def train_encoder(
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    next_obs: np.ndarray,
    latent_dim: int,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    alpha_reward: float,
    beta_var: float,
    seed: int,
    device: str,
    k_step: int = 1,
    gamma: float = 1.0,
    obs_seq: np.ndarray | None = None,
    act_seq: np.ndarray | None = None,
    rew_seq: np.ndarray | None = None,
) -> tuple[FrozenEncoder, FrozenEncoder, dict, dict, list[dict[str, float]]]:
    """Train the dynamics encoder and collect validation diagnostics.

    Args:
        obs: Flat raw observations at transition starts.
        act: Flat raw actions aligned with ``obs``.
        rew: Flat raw rewards aligned with ``obs``.
        next_obs: Flat raw next observations.
        latent_dim: Dimension of the learned latent representation.
        hidden: Hidden width for the encoder and prediction heads.
        batch_size: Number of examples per optimizer step.
        epochs: Number of training epochs.
        lr: AdamW learning rate.
        alpha_reward: Weight on reward prediction loss.
        beta_var: Weight on latent variance penalty.
        seed: Random seed for train/validation split and PyTorch.
        device: Torch device used for training.
        k_step: Number of rollout steps to train over.
        gamma: Per-step decay for K-step dynamics and reward losses.
        obs_seq: Optional K-step observation windows.
        act_seq: Optional K-step action windows.
        rew_seq: Optional K-step reward windows.

    Returns:
        Last and best frozen encoders, their diagnostics, and training history.
    """
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma must be in (0, 1].")
    if k_step > 1 and (obs_seq is None or act_seq is None or rew_seq is None):
        raise ValueError("K-step training requires obs_seq, act_seq, and rew_seq.")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_scaler = StandardScaler()
    act_scaler = StandardScaler()

    obs_n = obs_scaler.fit_transform(obs).astype(np.float32)
    next_obs_n = obs_scaler.transform(next_obs).astype(np.float32)
    act_n = act_scaler.fit_transform(act).astype(np.float32)

    # Reward normalization is useful for numerical stability.
    rew_mean = float(rew.mean())
    rew_std = float(rew.std() + 1e-8)
    rew_n = ((rew - rew_mean) / rew_std).astype(np.float32)

    if k_step == 1:
        n_examples = len(obs)
    else:
        obs_seq_shape = obs_seq.shape
        act_seq_shape = act_seq.shape
        obs_seq_n = obs_scaler.transform(obs_seq.reshape(-1, obs.shape[1])).astype(np.float32)
        obs_seq_n = obs_seq_n.reshape(obs_seq_shape)
        act_seq_n = act_scaler.transform(act_seq.reshape(-1, act.shape[1])).astype(np.float32)
        act_seq_n = act_seq_n.reshape(act_seq_shape)
        rew_seq_n = ((rew_seq - rew_mean) / rew_std).astype(np.float32)
        n_examples = len(obs_seq_n)

    indices = rng.permutation(n_examples)
    n_val = max(1, int(0.1 * len(indices)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) < batch_size:
        train_idx = indices
        val_idx = indices

    if k_step == 1:
        train_loader = make_transition_loader(
            obs_n=obs_n,
            act_n=act_n,
            rew_n=rew_n,
            next_obs_n=next_obs_n,
            indices=train_idx,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = make_transition_loader(
            obs_n=obs_n,
            act_n=act_n,
            rew_n=rew_n,
            next_obs_n=next_obs_n,
            indices=val_idx,
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        train_loader = make_k_step_transition_loader(
            obs_seq_n=obs_seq_n,
            act_seq_n=act_seq_n,
            rew_seq_n=rew_seq_n,
            indices=train_idx,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = make_k_step_transition_loader(
            obs_seq_n=obs_seq_n,
            act_seq_n=act_seq_n,
            rew_seq_n=rew_seq_n,
            indices=val_idx,
            batch_size=batch_size,
            shuffle=False,
        )

    obs_dim = obs.shape[1]
    act_dim = act.shape[1]

    model = DynamicsEncoder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"loss": 0.0, "dyn": 0.0, "dyn_rel": 0.0, "rew": 0.0, "var": 0.0}
        n_batches = 0

        for batch in train_loader:
            if k_step == 1:
                obs_b, act_b, rew_b, next_obs_b = batch
                losses = compute_batch_losses(
                    model=model,
                    obs_b=obs_b.to(device),
                    act_b=act_b.to(device),
                    rew_b=rew_b.to(device),
                    next_obs_b=next_obs_b.to(device),
                    alpha_reward=alpha_reward,
                    beta_var=beta_var,
                )
            else:
                obs_seq_b, act_seq_b, rew_seq_b = batch
                losses = compute_k_step_batch_losses(
                    model=model,
                    obs_seq_b=obs_seq_b.to(device),
                    act_seq_b=act_seq_b.to(device),
                    rew_seq_b=rew_seq_b.to(device),
                    alpha_reward=alpha_reward,
                    beta_var=beta_var,
                    gamma=gamma,
                )
            loss = losses["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            for name, value in losses.items():
                train_totals[name] += float(value.item())
            n_batches += 1

        train_metrics = {name: value / n_batches for name, value in train_totals.items()}
        val_metrics = evaluate_loader_losses(
            model=model,
            loader=val_loader,
            alpha_reward=alpha_reward,
            beta_var=beta_var,
            device=device,
            k_step=k_step,
            gamma=gamma,
        )
        row = {"epoch": float(epoch)}
        row.update({f"train_{name}": value for name, value in train_metrics.items()})
        row.update({f"val_{name}": value for name, value in val_metrics.items()})
        history.append(row)
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_epoch = epoch
            best_state_dict = freeze_model_state_dict(model)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_dyn={row['val_dyn']:.5f} | "
            f"val_rew={row['val_rew']:.5f} | "
            f"val_var={row['val_var']:.5f} | "
            f"val_dyn_rel={row['val_dyn_rel']:.5f}"
        )

    if best_state_dict is None:
        best_state_dict = freeze_model_state_dict(model)

    payload = FrozenEncoder(
        obs_scaler=obs_scaler,
        act_scaler=act_scaler,
        model_state_dict=freeze_model_state_dict(model),
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    )
    best_payload = FrozenEncoder(
        obs_scaler=obs_scaler,
        act_scaler=act_scaler,
        model_state_dict=best_state_dict,
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    )

    diagnostics = evaluate_encoder(
        model=model,
        obs_scaler=obs_scaler,
        act_scaler=act_scaler,
        rew_mean=rew_mean,
        rew_std=rew_std,
        obs=obs,
        act=act,
        rew=rew,
        next_obs=next_obs,
        device=device,
        seed=seed,
    )
    diagnostics["best_val_loss"] = best_val_loss
    diagnostics["best_epoch"] = float(best_epoch)

    best_model = best_payload.build_model().to(device)
    best_diagnostics = evaluate_encoder(
        model=best_model,
        obs_scaler=obs_scaler,
        act_scaler=act_scaler,
        rew_mean=rew_mean,
        rew_std=rew_std,
        obs=obs,
        act=act,
        rew=rew,
        next_obs=next_obs,
        device=device,
        seed=seed,
    )
    best_diagnostics["best_val_loss"] = best_val_loss
    best_diagnostics["best_epoch"] = float(best_epoch)

    return payload, best_payload, diagnostics, best_diagnostics, history


def freeze_model_state_dict(model: DynamicsEncoder) -> dict[str, torch.Tensor]:
    """Copy a model state dict into CPU tensors safe for checkpointing.

    Args:
        model: Dynamics encoder whose state should be snapshotted.

    Returns:
        A detached CPU copy of the model state dictionary.
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def main():
    """Train a Minari dynamics encoder from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=DATASET.keys(), default="Swimmer")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-reward", type=float, default=1.0)
    parser.add_argument("--beta-var", type=float, default=1.0)
    parser.add_argument("--k-step", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--max-neighborhood-points", type=int, default=10_000)
    parser.add_argument("--neighbor-values", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()
    if args.k_step < 1:
        raise ValueError("--k-step must be at least 1.")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1].")

    dataset_ids = DATASET[args.env]
    print(f"Using datasets: {dataset_ids}")

    # if args.env == 'HalfCheetah':
    args.lr = 5e-5

    output_dir = args.output_dir / f"{args.env}-v5" / f"nn_encoder_{args.k_step}_steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    datasets = load_minari_datasets(dataset_ids, force_download=args.force_download)
    transitions = flatten_transitions(datasets)
    k_step_windows = None
    if args.k_step > 1:
        k_step_windows = flatten_k_step_transitions(datasets, k_step=args.k_step)

    print(f"Transitions: {len(transitions.obs)}")
    print(f"obs_dim={transitions.obs.shape[1]}, act_dim={transitions.act.shape[1]}")
    if k_step_windows is not None:
        print(f"{args.k_step}-step windows: {len(k_step_windows.obs_seq)}")

    encoder, best_encoder, diagnostics, best_diagnostics, history = train_encoder(
        obs=transitions.obs,
        act=transitions.act,
        rew=transitions.rew,
        next_obs=transitions.next_obs,
        latent_dim=args.components,
        hidden=args.hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        alpha_reward=args.alpha_reward,
        beta_var=args.beta_var,
        seed=args.seed,
        device=args.device,
        k_step=args.k_step,
        gamma=args.gamma,
        obs_seq=k_step_windows.obs_seq if k_step_windows is not None else None,
        act_seq=k_step_windows.act_seq if k_step_windows is not None else None,
        rew_seq=k_step_windows.rew_seq if k_step_windows is not None else None,
    )

    output_path = output_dir / "dynamics_encoder.pkl"
    best_output_path = output_dir / "dynamics_encoder_best.pkl"

    model = encoder.build_model().to(args.device)
    z = encode_normalized_observations(
        model=model,
        obs=transitions.obs,
        obs_scaler=encoder.obs_scaler,
        device=args.device,
        batch_size=args.batch_size,
    )
    obs_n = encoder.obs_scaler.transform(transitions.obs).astype(np.float32)
    overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=obs_n,
        z=z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )

    for n_neighbors, overlaps in overlaps_by_k.items():
        diagnostics[f"{n_neighbors}nn_preservation"] = float(np.mean(overlaps))

    save_encoder(encoder, diagnostics, history, output_path)
    best_model = best_encoder.build_model().to(args.device)
    best_z = encode_normalized_observations(
        model=best_model,
        obs=transitions.obs,
        obs_scaler=best_encoder.obs_scaler,
        device=args.device,
        batch_size=args.batch_size,
    )
    best_obs_n = best_encoder.obs_scaler.transform(transitions.obs).astype(np.float32)
    best_overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=best_obs_n,
        z=best_z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )
    for n_neighbors, overlaps in best_overlaps_by_k.items():
        best_diagnostics[f"{n_neighbors}nn_preservation"] = float(np.mean(overlaps))

    save_encoder(best_encoder, best_diagnostics, history, best_output_path)
    plot_training_history(history, output_dir)
    plot_latent_scatter(
        z=z,
        labels=transitions.dataset_labels,
        episode_returns=transitions.episode_returns,
        time_in_episode=transitions.time_in_episode,
        dataset_ids=dataset_ids,
        output_dir=output_dir,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_latent_spectrum(z, output_dir)
    plot_neighborhood_preservation(
        overlaps_by_k=overlaps_by_k,
        output_dir=output_dir,
        obs_n=obs_n,
        z=z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )
    plot_latent_scatter(
        z=best_z,
        labels=transitions.dataset_labels,
        episode_returns=transitions.episode_returns,
        time_in_episode=transitions.time_in_episode,
        dataset_ids=dataset_ids,
        output_dir=output_dir,
        max_points=args.max_plot_points,
        seed=args.seed,
        filename_prefix="best_",
    )
    plot_latent_spectrum(best_z, output_dir, filename_prefix="best_")
    plot_neighborhood_preservation(
        overlaps_by_k=best_overlaps_by_k,
        output_dir=output_dir,
        obs_n=best_obs_n,
        z=best_z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
        filename_prefix="best_",
    )
    plot_encoder_neighborhood_comparison(
        left_overlaps_by_k=overlaps_by_k,
        right_overlaps_by_k=best_overlaps_by_k,
        output_dir=output_dir,
        output_name="best_vs_last_neighborhood",
        left_label="last encoder",
        right_label="best encoder",
    )

    print("Diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.6g}")

    print(f"Saved frozen encoder to: {output_path}")
    print(f"Saved best frozen encoder to: {best_output_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
