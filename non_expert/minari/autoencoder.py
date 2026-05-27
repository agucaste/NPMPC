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
    load_minari_datasets,
    neighborhood_preservation_score,
    plot_latent_scatter,
    plot_latent_spectrum,
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


@dataclass
class ObservationData:
    """Flat observation arrays plus labels for representation diagnostics."""

    obs: np.ndarray
    dataset_labels: np.ndarray
    episode_returns: np.ndarray
    time_in_episode: np.ndarray


def flatten_observations(datasets: dict[str, object]) -> ObservationData:
    """Flatten Minari episode observations into one observation table.

    Args:
        datasets: Mapping from Minari dataset id to loaded Minari dataset.

    Returns:
        Flat observations and per-observation metadata for diagnostics.
    """
    obs_list = []
    dataset_label_list = []
    episode_return_list = []
    time_in_episode_list = []

    for dataset_id, dataset in datasets.items():
        for episode in dataset.iterate_episodes():
            obs = np.asarray(episode.observations, dtype=np.float32)
            rew = np.asarray(episode.rewards, dtype=np.float32)
            episode_return = float(np.sum(rew))

            obs_list.append(obs)
            dataset_label_list.extend([dataset_id] * len(obs))
            episode_return_list.extend([episode_return] * len(obs))
            time_in_episode_list.extend(np.linspace(0.0, 1.0, len(obs), endpoint=True))

    return ObservationData(
        obs=np.concatenate(obs_list, axis=0),
        dataset_labels=np.asarray(dataset_label_list),
        episode_returns=np.asarray(episode_return_list, dtype=np.float32),
        time_in_episode=np.asarray(time_in_episode_list, dtype=np.float32),
    )


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
) -> tuple[FrozenAutoEncoder, dict, list[dict[str, float]]]:
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

        print(
            f"epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.5f} | "
            f"val_loss={row['val_loss']:.5f} | "
            f"val_rec={row['val_rec']:.5f} | "
            f"val_var={row['val_var']:.5f}"
        )

    payload = FrozenAutoEncoder(
        obs_scaler=obs_scaler,
        model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
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

    return payload, diagnostics, history


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
) -> None:
    """Plot raw-space reconstruction error and coordinate agreement."""
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

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].hist(reconstruction_errors, bins=60, color="C0", alpha=0.9)
    axs[0].set_title("Raw reconstruction error")
    axs[0].set_xlabel("||s - s_hat||2")
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

    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_quality.pdf")
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta-var", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--max-neighborhood-points", type=int, default=10_000)
    parser.add_argument("--neighbors", type=int, default=100)
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
    observations = flatten_observations(datasets)

    print(f"Observations: {len(observations.obs)}")
    print(f"obs_dim={observations.obs.shape[1]}")

    autoencoder, diagnostics, history = train_autoencoder(
        obs=observations.obs,
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

    model = autoencoder.build_model().to(args.device)
    z = encode_normalized_observations(
        model=model,
        obs=observations.obs,
        obs_scaler=autoencoder.obs_scaler,
        device=args.device,
        batch_size=args.batch_size,
    )
    obs_n = autoencoder.obs_scaler.transform(observations.obs).astype(np.float32)
    diagnostics[f"{args.neighbors}nn_preservation"] = neighborhood_preservation_score(
        obs_n=obs_n,
        z=z,
        max_points=args.max_neighborhood_points,
        n_neighbors=args.neighbors,
        seed=args.seed,
    )

    save_autoencoder(autoencoder, diagnostics, history, output_path)
    plot_training_history(history, output_dir)
    plot_reconstruction_quality(
        model=model,
        obs=observations.obs,
        obs_scaler=autoencoder.obs_scaler,
        output_dir=output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_latent_scatter(
        z=z,
        labels=observations.dataset_labels,
        episode_returns=observations.episode_returns,
        time_in_episode=observations.time_in_episode,
        dataset_ids=dataset_ids,
        output_dir=output_dir,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_latent_spectrum(z, output_dir)

    print("Diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.6g}")

    print(f"Saved frozen autoencoder to: {output_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
