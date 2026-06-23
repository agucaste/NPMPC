"""
Deterministic Minari autoencoder baseline.

This trains a nonlinear observation compressor:
    z = phi(s)
    s_hat = decoder(z)

The model operates on StandardScaler-normalized observations. After training,
only the encoder is needed for downstream observation featurization.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from non_expert.minari.utils import (
    DATASET,
    MLP,
    encode_normalized_observations,
    flatten_transitions,
    load_minari_datasets,
    neighborhood_preservation_overlaps,
    plot_encoder_neighborhood_comparison,
    plot_latent_scatter,
    plot_latent_spectrum,
    plot_neighborhood_preservation,
    sample_indices,
    variance_loss,
)


class AutoEncoder(nn.Module):
    """Observation autoencoder with an MLP encoder and decoder."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden: int = 256):
        """Create the autoencoder modules.

        Args:
            obs_dim: Dimension of normalized observations.
            latent_dim: Dimension of the learned latent representation.
            hidden: Hidden width used by the MLP encoder and decoder.
        """
        super().__init__()
        self.encoder = MLP(obs_dim, latent_dim, hidden=hidden)
        self.decoder = MLP(latent_dim, obs_dim, hidden=hidden)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Map normalized observations to latent vectors."""
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct normalized observations through the latent bottleneck."""
        z = self.encode(obs)
        obs_hat = self.decoder(z)
        return z, obs_hat


@dataclass
class FrozenAutoEncoder:
    """Pickle-friendly frozen autoencoder artifact."""

    obs_scaler: StandardScaler
    model_state_dict: dict
    obs_dim: int
    latent_dim: int
    hidden: int

    def build_model(self) -> AutoEncoder:
        """Reconstruct the PyTorch module in evaluation mode."""
        model = AutoEncoder(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
        )
        model.load_state_dict(self.model_state_dict)
        model.eval()
        return model

    @torch.no_grad()
    def encode(self, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Encode one raw observation or a batch of raw observations."""
        obs = np.asarray(obs, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]

        obs_n = self.obs_scaler.transform(obs).astype(np.float32)

        model = self.build_model().to(device)
        x = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
        z = model.encode(x).cpu().numpy()

        return z[0] if single else z


def make_observation_loader(
    obs_n: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader over normalized observations."""
    dataset = TensorDataset(torch.from_numpy(obs_n[indices]))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def compute_batch_losses(
    model: AutoEncoder,
    obs_b: torch.Tensor,
    beta_var: float,
) -> dict[str, torch.Tensor]:
    """Compute autoencoder losses for one observation batch."""
    z, obs_hat = model(obs_b)
    rec_loss = torch.mean((obs_hat - obs_b) ** 2)
    var_loss = variance_loss(z)
    loss = rec_loss + beta_var * var_loss
    return {
        "loss": loss,
        "rec": rec_loss,
        "var": var_loss,
    }


@torch.no_grad()
def evaluate_loader_losses(
    model: AutoEncoder,
    loader: DataLoader,
    beta_var: float,
    device: str,
) -> dict[str, float]:
    """Average autoencoder losses over a DataLoader."""
    model.eval()
    totals = {"loss": 0.0, "rec": 0.0, "var": 0.0}
    n_batches = 0

    for (obs_b,) in loader:
        losses = compute_batch_losses(
            model=model,
            obs_b=obs_b.to(device),
            beta_var=beta_var,
        )
        for name, value in losses.items():
            totals[name] += float(value.item())
        n_batches += 1

    if n_batches == 0:
        return {name: float("nan") for name in totals}
    return {name: value / n_batches for name, value in totals.items()}


def train_autoencoder(
    obs: np.ndarray,
    latent_dim: int,
    hidden: int,
    batch_size: int,
    epochs: int,
    lr: float,
    beta_var: float,
    seed: int,
    device: str,
) -> tuple[FrozenAutoEncoder, FrozenAutoEncoder, dict, dict, list[dict[str, float]]]:
    """Train the autoencoder and collect validation diagnostics."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs_scaler = StandardScaler()
    obs_n = obs_scaler.fit_transform(obs).astype(np.float32)

    indices = rng.permutation(len(obs))
    n_val = max(1, int(0.1 * len(indices)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) < batch_size:
        train_idx = indices
        val_idx = indices

    train_loader = make_observation_loader(
        obs_n=obs_n,
        indices=train_idx,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = make_observation_loader(
        obs_n=obs_n,
        indices=val_idx,
        batch_size=batch_size,
        shuffle=False,
    )

    obs_dim = obs.shape[1]
    model = AutoEncoder(obs_dim=obs_dim, latent_dim=latent_dim, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"loss": 0.0, "rec": 0.0, "var": 0.0}
        n_batches = 0

        for (obs_b,) in train_loader:
            losses = compute_batch_losses(
                model=model,
                obs_b=obs_b.to(device),
                beta_var=beta_var,
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
            beta_var=beta_var,
            device=device,
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
            f"val_rec={row['val_rec']:.5f} | "
            f"val_var={row['val_var']:.5f}"
        )

    if best_state_dict is None:
        best_state_dict = freeze_model_state_dict(model)

    payload = FrozenAutoEncoder(
        obs_scaler=obs_scaler,
        model_state_dict=freeze_model_state_dict(model),
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    )
    best_payload = FrozenAutoEncoder(
        obs_scaler=obs_scaler,
        model_state_dict=best_state_dict,
        obs_dim=obs_dim,
        latent_dim=latent_dim,
        hidden=hidden,
    )

    diagnostics = evaluate_autoencoder(
        model=model,
        obs_scaler=obs_scaler,
        obs=obs,
        device=device,
        seed=seed,
    )
    diagnostics["best_val_loss"] = best_val_loss
    diagnostics["best_epoch"] = float(best_epoch)

    best_model = best_payload.build_model().to(device)
    best_diagnostics = evaluate_autoencoder(
        model=best_model,
        obs_scaler=obs_scaler,
        obs=obs,
        device=device,
        seed=seed,
    )
    best_diagnostics["best_val_loss"] = best_val_loss
    best_diagnostics["best_epoch"] = float(best_epoch)

    return payload, best_payload, diagnostics, best_diagnostics, history


def freeze_model_state_dict(model: AutoEncoder) -> dict[str, torch.Tensor]:
    """Copy an autoencoder state dict into CPU tensors for checkpointing.

    Args:
        model: Autoencoder whose parameters should be snapshotted.

    Returns:
        A detached CPU copy of the model state dictionary.
    """
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate_autoencoder(
    model: AutoEncoder,
    obs_scaler: StandardScaler,
    obs: np.ndarray,
    device: str,
    max_eval_points: int = 50_000,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate reconstruction quality and latent usage diagnostics."""
    model.eval()

    n = min(len(obs), max_eval_points)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obs), size=n, replace=False)

    obs_raw = obs[idx]
    obs_n = obs_scaler.transform(obs_raw).astype(np.float32)
    obs_t = torch.as_tensor(obs_n, dtype=torch.float32, device=device)
    z, obs_hat_t = model(obs_t)
    obs_hat_n = obs_hat_t.cpu().numpy()
    obs_raw_from_n = obs_scaler.inverse_transform(obs_n)
    obs_hat_raw = obs_scaler.inverse_transform(obs_hat_n)

    rec_mse_normalized = float(np.mean((obs_hat_n - obs_n) ** 2))
    rec_mse_raw = float(np.mean((obs_hat_raw - obs_raw_from_n) ** 2))

    z_np = z.cpu().numpy()
    latent_std = np.std(z_np, axis=0)
    cov = np.atleast_2d(np.cov(z_np, rowvar=False))
    eigvals = np.linalg.eigvalsh(cov)

    return {
        "reconstruction_mse_normalized": rec_mse_normalized,
        "reconstruction_mse_raw": rec_mse_raw,
        "latent_std_min": float(latent_std.min()),
        "latent_std_median": float(np.median(latent_std)),
        "latent_std_max": float(latent_std.max()),
        "latent_cov_eig_min": float(eigvals.min()),
        "latent_cov_eig_median": float(np.median(eigvals)),
        "latent_cov_eig_max": float(eigvals.max()),
    }


def plot_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """Plot train and validation curves for the autoencoder losses."""
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in history])
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    specs = [
        ("loss", "Total loss"),
        ("rec", "Reconstruction MSE"),
        ("var", "Variance penalty"),
    ]

    for ax, (key, title) in zip(axs.ravel(), specs):
        ax.plot(epochs, [row[f"train_{key}"] for row in history], label="train")
        ax.plot(epochs, [row[f"val_{key}"] for row in history], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


@torch.no_grad()
def plot_reconstruction_quality(
    model: AutoEncoder,
    obs: np.ndarray,
    obs_scaler: StandardScaler,
    output_dir: Path,
    device: str,
    batch_size: int,
    max_points: int,
    seed: int,
    filename_prefix: str = "",
) -> None:
    """Plot normalized raw-space reconstruction error and coordinate agreement."""
    import matplotlib.pyplot as plt

    model.eval()
    rng = np.random.default_rng(seed)
    idx = sample_indices(obs.shape[0], max_points=max_points, rng=rng)
    obs_raw = obs[idx]
    obs_n = obs_scaler.transform(obs_raw).astype(np.float32)

    chunks = []
    for start in range(0, len(obs_n), batch_size):
        batch = torch.as_tensor(obs_n[start : start + batch_size], dtype=torch.float32, device=device)
        _, obs_hat = model(batch)
        chunks.append(obs_hat.cpu().numpy())

    obs_hat_n = np.concatenate(chunks, axis=0)
    obs_hat_raw = obs_scaler.inverse_transform(obs_hat_n)
    reconstruction_errors = np.linalg.norm(obs_raw - obs_hat_raw, axis=1)
    obs_norms = np.linalg.norm(obs_raw, axis=1)
    normalized_reconstruction_errors = reconstruction_errors / np.maximum(obs_norms, 1e-12)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(normalized_reconstruction_errors, bins=60, color="C0", alpha=0.9)
    axs[0].set_title("Normalized reconstruction error")
    axs[0].set_xlabel("||s - s_hat||2 / ||s||2")
    axs[0].set_ylabel("count")
    axs[0].grid(alpha=0.25)

    raw_values = obs_raw.ravel()
    recon_values = obs_hat_raw.ravel()
    max_scatter_values = 100_000
    if raw_values.size > max_scatter_values:
        value_idx = rng.choice(raw_values.size, size=max_scatter_values, replace=False)
        raw_values = raw_values[value_idx]
        recon_values = recon_values[value_idx]

    axs[1].scatter(raw_values, recon_values, s=3, alpha=0.2, linewidths=0)
    vmin = float(min(raw_values.min(), recon_values.min()))
    vmax = float(max(raw_values.max(), recon_values.max()))
    axs[1].plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1.0, alpha=0.7)
    axs[1].set_title("Original vs reconstructed coordinates")
    axs[1].set_xlabel("original")
    axs[1].set_ylabel("reconstructed")
    axs[1].grid(alpha=0.25)

    n_sample = len(idx)
    if n_sample == obs.shape[0]:
        sample_text = f"N={n_sample:,} transitions"
    else:
        sample_text = f"N={n_sample:,} sampled transitions from {obs.shape[0]:,}"
    fig.suptitle(f"Reconstruction diagnostics over {sample_text}")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_dir / f"{filename_prefix}reconstruction_quality.pdf")
    plt.close(fig)


def save_autoencoder(
    payload: FrozenAutoEncoder,
    diagnostics: dict,
    history: list[dict[str, float]],
    path: Path,
) -> None:
    """Save the frozen autoencoder and diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "type": "autoencoder",
                "obs_scaler": payload.obs_scaler,
                "model_state_dict": payload.model_state_dict,
                "obs_dim": payload.obs_dim,
                "latent_dim": payload.latent_dim,
                "hidden": payload.hidden,
                "diagnostics": diagnostics,
                "history": history,
            },
            f,
        )


def main() -> None:
    """Train a Minari autoencoder from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=DATASET.keys(), default="Swimmer")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta-var", type=float, default=1.0)
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

    dataset_ids = DATASET[args.env]
    print(f"Using datasets: {dataset_ids}")

    output_dir = args.output_dir / f"{args.env}-v5" / "autoencoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    datasets = load_minari_datasets(dataset_ids, force_download=args.force_download)
    transitions = flatten_transitions(datasets)

    print(f"Transitions: {len(transitions.obs)}")
    print(f"obs_dim={transitions.obs.shape[1]}, act_dim={transitions.act.shape[1]}")

    autoencoder, best_autoencoder, diagnostics, best_diagnostics, history = train_autoencoder(
        obs=transitions.obs,
        latent_dim=args.components,
        hidden=args.hidden,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta_var=args.beta_var,
        seed=args.seed,
        device=args.device,
    )

    output_path = output_dir / "autoencoder.pkl"
    best_output_path = output_dir / "autoencoder_best.pkl"

    model = autoencoder.build_model().to(args.device)
    z = encode_normalized_observations(
        model=model,
        obs=transitions.obs,
        obs_scaler=autoencoder.obs_scaler,
        device=args.device,
        batch_size=args.batch_size,
    )
    obs_n = autoencoder.obs_scaler.transform(transitions.obs).astype(np.float32)
    overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=obs_n,
        z=z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )
    for n_neighbors, overlaps in overlaps_by_k.items():
        diagnostics[f"{n_neighbors}nn_preservation"] = float(np.mean(overlaps))

    save_autoencoder(autoencoder, diagnostics, history, output_path)
    best_model = best_autoencoder.build_model().to(args.device)
    best_z = encode_normalized_observations(
        model=best_model,
        obs=transitions.obs,
        obs_scaler=best_autoencoder.obs_scaler,
        device=args.device,
        batch_size=args.batch_size,
    )
    best_obs_n = best_autoencoder.obs_scaler.transform(transitions.obs).astype(np.float32)
    best_overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=best_obs_n,
        z=best_z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )
    for n_neighbors, overlaps in best_overlaps_by_k.items():
        best_diagnostics[f"{n_neighbors}nn_preservation"] = float(np.mean(overlaps))

    save_autoencoder(best_autoencoder, best_diagnostics, history, best_output_path)
    plot_training_history(history, output_dir)
    plot_reconstruction_quality(
        model=model,
        obs=transitions.obs,
        obs_scaler=autoencoder.obs_scaler,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
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
    plot_reconstruction_quality(
        model=best_model,
        obs=transitions.obs,
        obs_scaler=best_autoencoder.obs_scaler,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_points=args.max_plot_points,
        seed=args.seed,
        filename_prefix="best_",
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
        left_label="last autoencoder",
        right_label="best autoencoder",
    )

    print("Diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.6g}")

    print(f"Saved frozen autoencoder to: {output_path}")
    print(f"Saved best frozen autoencoder to: {best_output_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
