"""
Minimal Minari encoder pretraining example.

Learns an encoder phi(s) so that:
    z_t = phi(s_t)
    z_t + g(z_t, a_t) ~= phi(s_{t+1})
    h(z_t, a_t) ~= r_t

After training, discard g and h. Keep phi frozen.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import minari
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from non_expert.minari.dynamics_encoder_model import DynamicsEncoder


DATASET = {
    "InvertedPendulum": (
        "mujoco/invertedpendulum/medium-v0",
        "mujoco/invertedpendulum/expert-v0",
    ),
    "HalfCheetah": (
        "mujoco/halfcheetah/medium-v0",
        "mujoco/halfcheetah/expert-v0",
    ),
    "Swimmer": (
        "mujoco/swimmer/medium-v0",
        "mujoco/swimmer/expert-v0",
    ),
}


@dataclass
class FrozenEncoder:
    """Pickle-friendly frozen observation encoder artifact."""

    obs_scaler: StandardScaler
    act_scaler: StandardScaler
    model_state_dict: dict
    obs_dim: int
    act_dim: int
    latent_dim: int
    hidden: int

    def build_model(self) -> DynamicsEncoder:
        """Reconstruct the PyTorch module in evaluation mode."""
        model = DynamicsEncoder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
        )
        model.load_state_dict(self.model_state_dict)
        model.eval()
        return model

    @torch.no_grad()
    def encode(self, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
        """Encode one observation or a batch of observations."""
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
class TransitionData:
    """Flat transition arrays plus labels for representation diagnostics."""

    obs: np.ndarray
    act: np.ndarray
    rew: np.ndarray
    next_obs: np.ndarray
    dataset_labels: np.ndarray
    episode_returns: np.ndarray
    time_in_episode: np.ndarray


def load_minari_datasets(dataset_ids: tuple[str, ...], force_download: bool = False):
    """Load Minari datasets, downloading them if needed."""
    datasets = {}
    for dataset_id in dataset_ids:
        if force_download:
            minari.download_dataset(dataset_id, force_download=True)
        datasets[dataset_id] = minari.load_dataset(dataset_id, download=True)
    return datasets


def flatten_transitions(datasets: dict[str, object]) -> TransitionData:
    """Flatten Minari episodes into one transition table."""
    obs_list = []
    act_list = []
    rew_list = []
    next_obs_list = []
    dataset_label_list = []
    episode_return_list = []
    time_in_episode_list = []

    for dataset_id, dataset in datasets.items():
        for episode in dataset.iterate_episodes():
            obs = np.asarray(episode.observations, dtype=np.float32)
            act = np.asarray(episode.actions, dtype=np.float32)
            rew = np.asarray(episode.rewards, dtype=np.float32)

            # Minari convention: observations has length T+1, actions/rewards length T.
            obs_t = obs[:-1]
            obs_tp1 = obs[1:]

            if len(obs_t) != len(act) or len(act) != len(rew):
                raise ValueError(
                    f"Inconsistent episode lengths: "
                    f"obs_t={len(obs_t)}, act={len(act)}, rew={len(rew)}"
                )

            obs_list.append(obs_t)
            act_list.append(act)
            rew_list.append(rew)
            next_obs_list.append(obs_tp1)
            dataset_label_list.extend([dataset_id] * len(act))
            episode_return_list.extend([float(np.sum(rew))] * len(act))
            time_in_episode_list.extend(np.linspace(0.0, 1.0, len(act), endpoint=False))

    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    rew = np.concatenate(rew_list, axis=0)
    next_obs = np.concatenate(next_obs_list, axis=0)

    return TransitionData(
        obs=obs,
        act=act,
        rew=rew,
        next_obs=next_obs,
        dataset_labels=np.asarray(dataset_label_list),
        episode_returns=np.asarray(episode_return_list, dtype=np.float32),
        time_in_episode=np.asarray(time_in_episode_list, dtype=np.float32),
    )


def variance_loss(z: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
    """
    Prevents collapse phi(s) = constant.

    Penalizes latent coordinates whose minibatch std is below min_std.
    """
    std = torch.sqrt(z.var(dim=0) + 1e-6)
    return torch.mean(torch.relu(min_std - std) ** 2)


def sample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    """Return a reproducible subset of row indices."""
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def make_transition_loader(
    obs_n: np.ndarray,
    act_n: np.ndarray,
    rew_n: np.ndarray,
    next_obs_n: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader over normalized transition arrays."""
    dataset = TensorDataset(
        torch.from_numpy(obs_n[indices]),
        torch.from_numpy(act_n[indices]),
        torch.from_numpy(rew_n[indices]),
        torch.from_numpy(next_obs_n[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def compute_batch_losses(
    model: DynamicsEncoder,
    obs_b: torch.Tensor,
    act_b: torch.Tensor,
    rew_b: torch.Tensor,
    next_obs_b: torch.Tensor,
    alpha_reward: float,
    beta_var: float,
) -> dict[str, torch.Tensor]:
    """Compute the encoder pretraining losses for one batch."""
    z, z_next_pred, r_pred = model(obs_b, act_b)

    # Target embedding. Detach avoids chasing a moving target on both sides.
    with torch.no_grad():
        z_next = model.encode(next_obs_b)

    dyn_loss = torch.mean((z_next_pred - z_next) ** 2)
    # Persistence baseline: predict z_{t+1} as z_t.
    baseline_dyn_loss = torch.mean((z - z_next) ** 2)
    relative_dyn_loss = dyn_loss / torch.clamp(baseline_dyn_loss, min=1e-12)
    rew_loss = torch.mean((r_pred - rew_b) ** 2)
    var_loss = variance_loss(z)
    loss = dyn_loss + alpha_reward * rew_loss + beta_var * var_loss

    return {
        "loss": loss,
        "dyn": dyn_loss,
        "dyn_rel": relative_dyn_loss,
        "rew": rew_loss,
        "var": var_loss,
    }
    return {"loss": loss, "dyn": dyn_loss, "rew": rew_loss, "var": var_loss}


@torch.no_grad()
def evaluate_loader_losses(
    model: DynamicsEncoder,
    loader: DataLoader,
    alpha_reward: float,
    beta_var: float,
    device: str,
) -> dict[str, float]:
    """Average pretraining losses over a DataLoader."""
    model.eval()
    totals = {"loss": 0.0, "dyn": 0.0, "dyn_rel": 0.0, "rew": 0.0, "var": 0.0}
    n_batches = 0

    for obs_b, act_b, rew_b, next_obs_b in loader:
        losses = compute_batch_losses(
            model=model,
            obs_b=obs_b.to(device),
            act_b=act_b.to(device),
            rew_b=rew_b.to(device),
            next_obs_b=next_obs_b.to(device),
            alpha_reward=alpha_reward,
            beta_var=beta_var,
        )
        for name, value in losses.items():
            totals[name] += float(value.item())
        n_batches += 1

    if n_batches == 0:
        return {name: float("nan") for name in totals}
    return {name: value / n_batches for name, value in totals.items()}


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
) -> tuple[FrozenEncoder, dict, list[dict[str, float]]]:
    """Train the dynamics encoder and collect validation diagnostics."""
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

    indices = rng.permutation(len(obs))
    n_val = max(1, int(0.1 * len(indices)))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) < batch_size:
        train_idx = indices
        val_idx = indices

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

    for epoch in range(1, epochs + 1):
        model.train()
        train_totals = {"loss": 0.0, "dyn": 0.0, "dyn_rel": 0.0, "rew": 0.0, "var": 0.0}
        n_batches = 0

        for obs_b, act_b, rew_b, next_obs_b in train_loader:
            losses = compute_batch_losses(
                model=model,
                obs_b=obs_b.to(device),
                act_b=act_b.to(device),
                rew_b=rew_b.to(device),
                next_obs_b=next_obs_b.to(device),
                alpha_reward=alpha_reward,
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
            alpha_reward=alpha_reward,
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
            f"val_dyn={row['val_dyn']:.5f} | "
            f"val_rew={row['val_rew']:.5f} | "
            f"val_var={row['val_var']:.5f} | "
            f"val_dyn_rel={row['val_dyn_rel']:.5f}"
        )

    payload = FrozenEncoder(
        obs_scaler=obs_scaler,
        act_scaler=act_scaler,
        model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
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

    return payload, diagnostics, history


@torch.no_grad()
def evaluate_encoder(
    model: DynamicsEncoder,
    obs_scaler: StandardScaler,
    act_scaler: StandardScaler,
    rew_mean: float,
    rew_std: float,
    obs: np.ndarray,
    act: np.ndarray,
    rew: np.ndarray,
    next_obs: np.ndarray,
    device: str,
    max_eval_points: int = 50_000,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate final representation and auxiliary prediction quality."""
    model.eval()

    n = min(len(obs), max_eval_points)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obs), size=n, replace=False)

    obs_n = obs_scaler.transform(obs[idx]).astype(np.float32)
    next_obs_n = obs_scaler.transform(next_obs[idx]).astype(np.float32)
    act_n = act_scaler.transform(act[idx]).astype(np.float32)
    rew_n = ((rew[idx] - rew_mean) / rew_std).astype(np.float32)

    obs_t = torch.as_tensor(obs_n, device=device)
    next_obs_t = torch.as_tensor(next_obs_n, device=device)
    act_t = torch.as_tensor(act_n, device=device)
    rew_t = torch.as_tensor(rew_n, device=device)

    z, z_next_pred, r_pred = model(obs_t, act_t)
    z_next = model.encode(next_obs_t)

    dyn_mse = torch.mean((z_next_pred - z_next) ** 2).item()
    rew_mse = torch.mean((r_pred - rew_t) ** 2).item()
    rew_var = torch.var(rew_t).item()
    rew_r2 = 1.0 - rew_mse / max(rew_var, 1e-12)
    z_next_var = torch.var(z_next).item()
    dyn_r2 = 1.0 - dyn_mse / max(z_next_var, 1e-12)

    z_np = z.cpu().numpy()
    latent_std = np.std(z_np, axis=0)
    cov = np.atleast_2d(np.cov(z_np, rowvar=False))
    eigvals = np.linalg.eigvalsh(cov)

    return {
        "latent_dynamics_mse": dyn_mse,
        "latent_dynamics_r2": float(dyn_r2),
        "reward_mse_normalized": rew_mse,
        "reward_r2_normalized": float(rew_r2),
        "latent_std_min": float(latent_std.min()),
        "latent_std_median": float(np.median(latent_std)),
        "latent_std_max": float(latent_std.max()),
        "latent_cov_eig_min": float(eigvals.min()),
        "latent_cov_eig_median": float(np.median(eigvals)),
        "latent_cov_eig_max": float(eigvals.max()),
    }


@torch.no_grad()
def encode_normalized_observations(
    model: DynamicsEncoder,
    obs: np.ndarray,
    obs_scaler: StandardScaler,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Encode raw observations in batches."""
    model.eval()
    obs_n = obs_scaler.transform(obs).astype(np.float32)
    chunks = []
    for start in range(0, len(obs_n), batch_size):
        batch = torch.as_tensor(obs_n[start : start + batch_size], dtype=torch.float32, device=device)
        chunks.append(model.encode(batch).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def neighborhood_preservation_score(
    obs_n: np.ndarray,
    z: np.ndarray,
    max_points: int,
    n_neighbors: int,
    seed: int,
) -> float:
    """Measure how many nearest neighbors survive after encoding."""
    overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=obs_n,
        z=z,
        max_points=max_points,
        neighbor_values=[n_neighbors],
        seed=seed,
    )
    if not overlaps_by_k:
        return float("nan")
    overlaps = next(iter(overlaps_by_k.values()))
    return float(np.mean(overlaps))


def neighborhood_preservation_overlaps(
    obs_n: np.ndarray,
    z: np.ndarray,
    max_points: int,
    neighbor_values: list[int],
    seed: int,
) -> dict[int, np.ndarray]:
    """Compute per-point nearest-neighbor preservation overlaps.

    Args:
        obs_n: Normalized observations in the original observation space.
        z: Encoded observations in latent space.
        max_points: Maximum number of points sampled for nearest-neighbor comparison.
        neighbor_values: Neighborhood sizes to evaluate.
        seed: Random seed used when subsampling observations.

    Returns:
        Mapping from evaluated neighbor count to one overlap score per sampled point.
    """
    rng = np.random.default_rng(seed)
    idx = sample_indices(obs_n.shape[0], max_points=max_points, rng=rng)
    max_neighbors = len(idx) - 1
    if max_neighbors < 1:
        return {}

    valid_neighbors = sorted({min(k, max_neighbors) for k in neighbor_values if k > 0})
    if not valid_neighbors:
        return {}

    obs_sample = obs_n[idx]
    z_sample = z[idx]

    obs_order = nearest_neighbor_indices(obs_sample, max_neighbors=max(valid_neighbors))
    z_order = nearest_neighbor_indices(z_sample, max_neighbors=max(valid_neighbors))

    overlaps_by_k = {}
    for n_neighbors in valid_neighbors:
        obs_nn = obs_order[:, :n_neighbors]
        z_nn = z_order[:, :n_neighbors]
        overlaps_by_k[n_neighbors] = np.asarray(
            [
                len(set(obs_neighbors).intersection(z_neighbors)) / n_neighbors
                for obs_neighbors, z_neighbors in zip(obs_nn, z_nn)
            ],
            dtype=np.float32,
        )
    return overlaps_by_k


def nearest_neighbor_indices(points: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Find nearest-neighbor indices for each point, excluding the point itself.

    Args:
        points: Points used both as queries and as the neighbor reference set.
        max_neighbors: Number of neighbors to return for each point.

    Returns:
        Integer array with shape ``(n_points, max_neighbors)``.
    """
    points = np.asarray(points)
    n_points = points.shape[0]
    if max_neighbors >= n_points:
        raise ValueError("max_neighbors must be smaller than the number of points.")

    model = NearestNeighbors(n_neighbors=max_neighbors + 1)
    model.fit(points)
    indices = model.kneighbors(points, return_distance=False)
    neighbors = np.empty((n_points, max_neighbors), dtype=indices.dtype)
    for row_idx, row in enumerate(indices):
        neighbors[row_idx] = row[row != row_idx][:max_neighbors]
    return neighbors


def plot_neighborhood_preservation(overlaps_by_k: dict[int, np.ndarray], output_dir: Path) -> None:
    """Plot neighborhood-preservation overlap distributions."""
    if not overlaps_by_k:
        return

    neighbor_values = sorted(overlaps_by_k)
    distributions = [overlaps_by_k[k] for k in neighbor_values]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    parts = ax.violinplot(distributions, showmeans=False, showmedians=False, showextrema=False)
    violin_color = "C0"
    for body in parts["bodies"]:
        body.set_facecolor(violin_color)
        body.set_edgecolor("C0")
        body.set_alpha(0.9)
        body.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    for pos, (values, body) in enumerate(zip(distributions, parts["bodies"]), start=1):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        median_min, median_max = violin_bounds_at_y(body, median, fallback_center=pos)
        ax.hlines(median, median_min, median_max, color="#333333", linestyle="--", linewidth=1.0, zorder=4)
        for quartile in (q1, q3):
            line_min, line_max = violin_bounds_at_y(body, quartile, fallback_center=pos)
            ax.hlines(quartile, line_min, line_max, color="#333333", linestyle="-.", linewidth=0.9, zorder=4)

        if len(values) > 500:
            point_idx = rng.choice(len(values), size=500, replace=False)
            point_values = values[point_idx]
        else:
            point_values = values
        jitter = rng.normal(loc=0.0, scale=0.035, size=len(point_values))
        ax.scatter(
            np.full(len(point_values), pos) + jitter,
            point_values,
            color="black",
            alpha=0.18,
            s=6,
            linewidths=0.2,
            edgecolors="black",
            zorder=2,
        )

    ax.set_xticks(np.arange(1, len(neighbor_values) + 1))
    ax.set_xticklabels([f"{k}-NN" for k in neighbor_values])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("neighbor overlap")
    n_points = len(distributions[0])
    n_points_latex = f"{n_points:,}".replace(",", r"{,}")
    ax.set_title(
        rf"Neighborhood preservation between $\mathcal{{S}}$ and $\mathcal{{Z}}$, "
        rf"over $N={n_points_latex}$ points"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "neighborhood_preservation.pdf")
    plt.close(fig)


def violin_bounds_at_y(body, y_value: float, fallback_center: float) -> tuple[float, float]:
    """Find the horizontal extent of a violin body at a given y-value.

    Args:
        body: Matplotlib violin body returned by ``Axes.violinplot``.
        y_value: Vertical coordinate where the violin width should be measured.
        fallback_center: Center x-position used if the polygon has no crossing.

    Returns:
        Minimum and maximum x-coordinates of the violin polygon at ``y_value``.
    """
    vertices = body.get_paths()[0].vertices
    x_values = []
    for start, stop in zip(vertices, np.roll(vertices, -1, axis=0)):
        y0 = start[1]
        y1 = stop[1]
        if y0 == y1:
            if np.isclose(y_value, y0):
                x_values.extend([start[0], stop[0]])
            continue
        if min(y0, y1) <= y_value <= max(y0, y1):
            fraction = (y_value - y0) / (y1 - y0)
            x_values.append(start[0] + fraction * (stop[0] - start[0]))

    if len(x_values) < 2:
        return fallback_center, fallback_center

    padding = 0.01
    return min(x_values) + padding, max(x_values) - padding


def plot_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    """Plot train and validation curves for the auxiliary losses."""
    epochs = np.asarray([row["epoch"] for row in history])
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    specs = [
        ("loss", "Total loss"),
        ("dyn", "Latent dynamics MSE"),
        ("dyn_rel", "Relative dynamics error (↓)"),
        ("rew", "Reward MSE"),
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
    for ax in axs.ravel()[len(specs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def plot_latent_scatter(
    z: np.ndarray,
    labels: np.ndarray,
    episode_returns: np.ndarray,
    time_in_episode: np.ndarray,
    dataset_ids: tuple[str, ...],
    output_dir: Path,
    max_points: int,
    seed: int,
) -> None:
    """Plot the first two latent coordinates with useful colorings."""
    if z.shape[1] < 2:
        print("Skipping latent scatter because latent_dim < 2.")
        return

    rng = np.random.default_rng(seed)
    idx = sample_indices(z.shape[0], max_points=max_points, rng=rng)
    z_sample = z[idx]
    labels_sample = labels[idx]
    returns_sample = episode_returns[idx]
    time_sample = time_in_episode[idx]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharex=True, sharey=True)

    for dataset_id in dataset_ids:
        mask = labels_sample == dataset_id
        axs[0].scatter(
            z_sample[mask, 0],
            z_sample[mask, 1],
            s=7,
            alpha=0.35,
            label=dataset_id.split("/")[-1],
        )
    axs[0].set_title("Dataset")
    axs[0].legend(markerscale=2)

    positive_returns = returns_sample[returns_sample > 0]
    norm = None
    if positive_returns.size > 0 and positive_returns.max() > positive_returns.min():
        norm = LogNorm(vmin=positive_returns.min(), vmax=positive_returns.max())
    scatter = axs[1].scatter(
        z_sample[:, 0],
        z_sample[:, 1],
        c=returns_sample,
        s=7,
        alpha=0.35,
        cmap="viridis",
        norm=norm,
    )
    axs[1].set_title("Episode return")
    fig.colorbar(scatter, ax=axs[1], label="return")

    scatter = axs[2].scatter(
        z_sample[:, 0],
        z_sample[:, 1],
        c=time_sample,
        s=7,
        alpha=0.35,
        cmap="plasma",
    )
    axs[2].set_title("Progress through episode")
    fig.colorbar(scatter, ax=axs[2], label="fraction")

    for ax in axs:
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")

    fig.suptitle("Latent space diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_scatter.pdf")
    plt.close(fig)


def plot_latent_spectrum(z: np.ndarray, output_dir: Path) -> None:
    """Plot per-dimension latent standard deviations and covariance eigenvalues."""
    latent_std = np.std(z, axis=0)
    eigvals = np.linalg.eigvalsh(np.atleast_2d(np.cov(z, rowvar=False)))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].bar(np.arange(1, len(latent_std) + 1), latent_std)
    axs[0].set_title("Latent coordinate std")
    axs[0].set_xlabel("latent dimension")
    axs[0].set_ylabel("std")

    axs[1].plot(np.arange(1, len(eigvals) + 1), np.sort(eigvals)[::-1], marker="o")
    axs[1].set_title("Latent covariance spectrum")
    axs[1].set_xlabel("rank")
    axs[1].set_ylabel("eigenvalue")
    axs[1].set_yscale("log")

    fig.tight_layout()
    fig.savefig(output_dir / "latent_spectrum.pdf")
    plt.close(fig)


def save_encoder(payload: FrozenEncoder, diagnostics: dict, history: list[dict[str, float]], path: Path) -> None:
    """Save the frozen encoder and its diagnostics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "type": "dynamics",
                "obs_scaler": payload.obs_scaler,
                "act_scaler": payload.act_scaler,
                "model_state_dict": payload.model_state_dict,
                "obs_dim": payload.obs_dim,
                "act_dim": payload.act_dim,
                "latent_dim": payload.latent_dim,
                "hidden": payload.hidden,
                "diagnostics": diagnostics,
                "history": history,
            },
            f,
        )


def main():
    """Train a Minari dynamics encoder from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=DATASET.keys(), default="Swimmer")
    parser.add_argument("--components", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha-reward", type=float, default=1.0)
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

    if args.env == 'HalfCheetah':
        args.lr = 5e-5

    output_dir = args.output_dir / f"{args.env}-v5" / "nn_encoder"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    datasets = load_minari_datasets(dataset_ids, force_download=args.force_download)
    transitions = flatten_transitions(datasets)

    print(f"Transitions: {len(transitions.obs)}")
    print(f"obs_dim={transitions.obs.shape[1]}, act_dim={transitions.act.shape[1]}")

    encoder, diagnostics, history = train_encoder(
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
    )

    output_path = output_dir / "dynamics_encoder.pkl"

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
    )

    print("Diagnostics:")
    for k, v in diagnostics.items():
        print(f"  {k}: {v:.6g}")

    print(f"Saved frozen encoder to: {output_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
