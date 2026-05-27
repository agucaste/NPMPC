from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


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


class MLP(nn.Module):
    """Small feed-forward network used by the encoder and prediction heads."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 2):
        """Build an MLP with ReLU activations."""
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the network on a batch."""
        return self.net(x)


class DynamicsEncoder(nn.Module):
    """Encoder with latent dynamics and reward prediction heads."""

    def __init__(self, obs_dim: int, act_dim: int, latent_dim: int, hidden: int = 256):
        """Create the encoder and auxiliary heads."""
        super().__init__()
        self.encoder = MLP(obs_dim, latent_dim, hidden=hidden)
        self.dynamics_head = MLP(latent_dim + act_dim, latent_dim, hidden=hidden)
        self.reward_head = MLP(latent_dim + act_dim, 1, hidden=hidden)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Map normalized observations to latent vectors."""
        return self.encoder(obs)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next latent state and reward from an observation-action batch."""
        z = self.encode(obs)
        za = torch.cat([z, act], dim=-1)
        z_next_pred = z + self.dynamics_head(za)  # With this change, dynamics_head only predicts "Delta_z(zt, at), instead of the full next latent. May make learning easier."
        r_pred = self.reward_head(za).squeeze(-1)
        return z, z_next_pred, r_pred


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


@dataclass
class KStepTransitionData:
    """Episode-local K-step transition windows for latent rollout training."""

    obs_seq: np.ndarray
    act_seq: np.ndarray
    rew_seq: np.ndarray


def load_minari_datasets(dataset_ids: tuple[str, ...], force_download: bool = False):
    """Load Minari datasets, downloading them if needed."""
    import minari

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


def flatten_k_step_transitions(datasets: dict[str, object], k_step: int) -> KStepTransitionData:
    """Flatten Minari episodes into overlapping K-step transition windows.

    Args:
        datasets: Mapping from Minari dataset id to loaded Minari dataset.
        k_step: Number of latent rollout steps in each window.

    Returns:
        Overlapping episode-local windows with observation, action, and reward
        sequences aligned as ``obs[t:t+k+1]``, ``act[t:t+k]``, and
        ``rew[t:t+k]``.
    """
    if k_step < 1:
        raise ValueError("k_step must be at least 1.")

    obs_seq_list = []
    act_seq_list = []
    rew_seq_list = []

    for dataset in datasets.values():
        for episode in dataset.iterate_episodes():
            obs = np.asarray(episode.observations, dtype=np.float32)
            act = np.asarray(episode.actions, dtype=np.float32)
            rew = np.asarray(episode.rewards, dtype=np.float32)

            if len(obs) != len(act) + 1 or len(act) != len(rew):
                raise ValueError(
                    f"Inconsistent episode lengths: "
                    f"obs={len(obs)}, act={len(act)}, rew={len(rew)}"
                )

            n_windows = len(act) - k_step + 1
            if n_windows <= 0:
                continue

            for start in range(n_windows):
                stop = start + k_step
                obs_seq_list.append(obs[start : stop + 1])
                act_seq_list.append(act[start:stop])
                rew_seq_list.append(rew[start:stop])

    if not obs_seq_list:
        raise ValueError(f"No episodes contain at least {k_step} transitions.")

    return KStepTransitionData(
        obs_seq=np.stack(obs_seq_list, axis=0),
        act_seq=np.stack(act_seq_list, axis=0),
        rew_seq=np.stack(rew_seq_list, axis=0),
    )


def variance_loss(z: torch.Tensor, min_std: float = 0.1) -> torch.Tensor:
    """Penalize latent coordinates whose minibatch std is below ``min_std``."""
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


def make_k_step_transition_loader(
    obs_seq_n: np.ndarray,
    act_seq_n: np.ndarray,
    rew_seq_n: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader over normalized K-step transition windows."""
    validate_k_step_transition_arrays(obs_seq_n, act_seq_n, rew_seq_n, indices)
    dataset = TensorDataset(
        torch.from_numpy(obs_seq_n[indices]),
        torch.from_numpy(act_seq_n[indices]),
        torch.from_numpy(rew_seq_n[indices]),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)


def validate_k_step_transition_arrays(
    obs_seq: np.ndarray,
    act_seq: np.ndarray,
    rew_seq: np.ndarray,
    indices: np.ndarray,
) -> None:
    """Validate episode-local K-step window arrays before DataLoader creation.

    Args:
        obs_seq: Observation windows with shape ``(N, K + 1, obs_dim)``.
        act_seq: Action windows with shape ``(N, K, act_dim)``.
        rew_seq: Reward windows with shape ``(N, K)``.
        indices: Row indices selected for the DataLoader.

    Raises:
        ValueError: If the arrays cannot represent aligned episode-local
            K-step windows.
    """
    if obs_seq.ndim != 3:
        raise ValueError(f"obs_seq must have shape (N, K + 1, obs_dim), got {obs_seq.shape}.")
    if act_seq.ndim != 3:
        raise ValueError(f"act_seq must have shape (N, K, act_dim), got {act_seq.shape}.")
    if rew_seq.ndim != 2:
        raise ValueError(f"rew_seq must have shape (N, K), got {rew_seq.shape}.")
    if len(obs_seq) != len(act_seq) or len(act_seq) != len(rew_seq):
        raise ValueError(
            f"K-step window counts must match: "
            f"obs_seq={len(obs_seq)}, act_seq={len(act_seq)}, rew_seq={len(rew_seq)}."
        )
    if obs_seq.shape[1] != act_seq.shape[1] + 1:
        raise ValueError(
            f"Observation windows must have one more time step than actions: "
            f"obs_seq={obs_seq.shape}, act_seq={act_seq.shape}."
        )
    if rew_seq.shape[1] != act_seq.shape[1]:
        raise ValueError(
            f"Reward and action windows must have the same K dimension: "
            f"rew_seq={rew_seq.shape}, act_seq={act_seq.shape}."
        )
    if len(indices) and (indices.min() < 0 or indices.max() >= len(obs_seq)):
        raise ValueError(f"indices are out of bounds for {len(obs_seq)} K-step windows.")


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


def compute_k_step_batch_losses(
    model: DynamicsEncoder,
    obs_seq_b: torch.Tensor,
    act_seq_b: torch.Tensor,
    rew_seq_b: torch.Tensor,
    alpha_reward: float,
    beta_var: float,
) -> dict[str, torch.Tensor]:
    """Compute K-step latent rollout losses for one batch of windows."""
    k_step = act_seq_b.shape[1]
    z_hat = model.encode(obs_seq_b[:, 0])
    var_loss = variance_loss(z_hat)
    dyn_losses = []
    dyn_rel_losses = []
    rew_losses = []

    for step in range(k_step):
        act_k = act_seq_b[:, step]
        za = torch.cat([z_hat, act_k], dim=-1)
        z_next_hat = z_hat + model.dynamics_head(za)
        r_pred = model.reward_head(za).squeeze(-1)

        with torch.no_grad():
            z_prev_target = model.encode(obs_seq_b[:, step])
            z_target = model.encode(obs_seq_b[:, step + 1])

        dyn_loss_k = torch.mean((z_next_hat - z_target) ** 2)
        baseline_k = torch.mean((z_prev_target - z_target) ** 2)
        dyn_losses.append(dyn_loss_k)
        dyn_rel_losses.append(dyn_loss_k / torch.clamp(baseline_k, min=1e-12))
        rew_losses.append(torch.mean((r_pred - rew_seq_b[:, step]) ** 2))
        z_hat = z_next_hat

    dyn_loss = torch.stack(dyn_losses).mean()
    relative_dyn_loss = torch.stack(dyn_rel_losses).mean()
    rew_loss = torch.stack(rew_losses).mean()
    loss = dyn_loss + alpha_reward * rew_loss + beta_var * var_loss

    return {
        "loss": loss,
        "dyn": dyn_loss,
        "dyn_rel": relative_dyn_loss,
        "rew": rew_loss,
        "var": var_loss,
    }


@torch.no_grad()
def evaluate_loader_losses(
    model: DynamicsEncoder,
    loader: DataLoader,
    alpha_reward: float,
    beta_var: float,
    device: str,
    k_step: int = 1,
) -> dict[str, float]:
    """Average pretraining losses over a DataLoader."""
    model.eval()
    totals = {"loss": 0.0, "dyn": 0.0, "dyn_rel": 0.0, "rew": 0.0, "var": 0.0}
    n_batches = 0

    for batch in loader:
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
            )
        for name, value in losses.items():
            totals[name] += float(value.item())
        n_batches += 1

    if n_batches == 0:
        return {name: float("nan") for name in totals}
    return {name: value / n_batches for name, value in totals.items()}


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
    """Compute per-point nearest-neighbor preservation overlaps."""
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


def pca_neighborhood_preservation_overlaps(
    obs_n: np.ndarray,
    z: np.ndarray,
    max_points: int,
    neighbor_values: list[int],
    seed: int,
) -> tuple[dict[int, np.ndarray], int]:
    """Compute PCA nearest-neighbor preservation overlaps for comparison.

    Args:
        obs_n: Normalized observations in the original observation space.
        z: Learned latent representation used to choose PCA dimensionality.
        max_points: Maximum number of points sampled for nearest-neighbor comparison.
        neighbor_values: Neighborhood sizes to evaluate.
        seed: Random seed used when subsampling observations.

    Returns:
        PCA overlap scores by neighbor count and the PCA component count used.
    """
    latent_dim = int(z.shape[1])
    n_components = min(latent_dim, obs_n.shape[1], obs_n.shape[0])
    if n_components < 1:
        return {}, n_components

    pca = PCA(n_components=n_components)
    z_pca = pca.fit_transform(obs_n)
    return (
        neighborhood_preservation_overlaps(
            obs_n=obs_n,
            z=z_pca,
            max_points=max_points,
            neighbor_values=neighbor_values,
            seed=seed,
        ),
        n_components,
    )


def nearest_neighbor_indices(points: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Find nearest-neighbor indices for each point, excluding the point itself."""
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


def plot_neighborhood_preservation(
    overlaps_by_k: dict[int, np.ndarray],
    output_dir: Path,
    obs_n: np.ndarray | None = None,
    z: np.ndarray | None = None,
    max_points: int | None = None,
    neighbor_values: list[int] | None = None,
    seed: int = 0,
) -> None:
    """Plot learned and PCA neighborhood-preservation overlap distributions."""
    import matplotlib.pyplot as plt

    if not overlaps_by_k:
        return

    neighbor_values = sorted(overlaps_by_k)
    distributions = [overlaps_by_k[k] for k in neighbor_values]
    pca_overlaps_by_k = None
    pca_components = None
    latent_dim = None
    if obs_n is not None and z is not None:
        latent_dim = int(z.shape[1])
        pca_overlaps_by_k, pca_components = pca_neighborhood_preservation_overlaps(
            obs_n=obs_n,
            z=z,
            max_points=max_points if max_points is not None else obs_n.shape[0],
            neighbor_values=neighbor_values,
            seed=seed,
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(1, len(neighbor_values) + 1)
    learned_parts = ax.violinplot(
        distributions,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    format_split_violins(
        learned_parts["bodies"],
        side="left",
        facecolor="C0",
        edgecolor="C0",
    )

    pca_distributions = []
    pca_parts = None
    if pca_overlaps_by_k:
        pca_distributions = [pca_overlaps_by_k[k] for k in neighbor_values]
        pca_parts = ax.violinplot(
            pca_distributions,
            positions=positions,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        format_split_violins(
            pca_parts["bodies"],
            side="right",
            facecolor="C1",
            edgecolor="C1",
        )

    rng = np.random.default_rng(0)
    for pos, values, body in zip(positions, distributions, learned_parts["bodies"]):
        plot_split_violin_summary(
            ax=ax,
            values=values,
            body=body,
            position=pos,
            side="left",
            rng=rng,
        )
    if pca_parts is not None:
        for pos, values, body in zip(positions, pca_distributions, pca_parts["bodies"]):
            plot_split_violin_summary(
                ax=ax,
                values=values,
                body=body,
                position=pos,
                side="right",
                rng=rng,
            )

    if pca_parts is not None:
        ax.legend(
            [learned_parts["bodies"][0], pca_parts["bodies"][0]],
            ["learned encoder", "PCA"],
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{k}-NN" for k in neighbor_values])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("neighbor overlap")
    n_points = len(distributions[0])
    n_points_latex = f"{n_points:,}".replace(",", r"{,}")
    title = rf"Neighborhood preservation over $N={n_points_latex}$ points"
    if latent_dim is not None and pca_components is not None:
        if latent_dim == pca_components:
            title += rf", latent/PCA components={latent_dim}"
        else:
            title += rf", latent dim={latent_dim}, PCA components={pca_components}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "neighborhood_preservation.pdf")
    plt.close(fig)


def format_split_violins(bodies: list, side: str, facecolor: str, edgecolor: str) -> None:
    """Clip violin bodies to the requested side of their center line."""
    for body in bodies:
        vertices = body.get_paths()[0].vertices
        center = np.mean(vertices[:, 0])
        if side == "left":
            vertices[:, 0] = np.clip(vertices[:, 0], -np.inf, center)
        elif side == "right":
            vertices[:, 0] = np.clip(vertices[:, 0], center, np.inf)
        else:
            raise ValueError(f"Unknown split violin side: {side}")
        body.set_facecolor(facecolor)
        body.set_edgecolor(edgecolor)
        body.set_alpha(0.85)
        body.set_linewidth(1.2)


def plot_split_violin_summary(
    ax,
    values: np.ndarray,
    body,
    position: float,
    side: str,
    rng: np.random.Generator,
) -> None:
    """Draw quartile markers and sampled points on one split violin half."""
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
    median_min, median_max = violin_bounds_at_y(body, median, fallback_center=position)
    ax.hlines(median, median_min, median_max, color="#333333", linestyle="--", linewidth=1.0, zorder=4)
    for quartile in (q1, q3):
        line_min, line_max = violin_bounds_at_y(body, quartile, fallback_center=position)
        ax.hlines(quartile, line_min, line_max, color="#333333", linestyle="-.", linewidth=0.9, zorder=4)

    point_values = values
    if len(values) > 500:
        point_idx = rng.choice(len(values), size=500, replace=False)
        point_values = values[point_idx]

    direction = -1.0 if side == "left" else 1.0
    jitter = direction * np.abs(rng.normal(loc=0.03, scale=0.025, size=len(point_values)))
    ax.scatter(
        np.full(len(point_values), position) + jitter,
        point_values,
        color="#495AF5" if side == "left" else "#F18A3B",  # Blueish or orange, to mimic C0/C1
        alpha=0.15,
        s=6,
        linewidths=0.2,
        edgecolors="black",
        zorder=2,
    )


def violin_bounds_at_y(body, y_value: float, fallback_center: float) -> tuple[float, float]:
    """Find the horizontal extent of a violin body at a given y-value."""
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
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] for row in history])
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    specs = [
        ("loss", "Total loss"),
        ("dyn", "Latent dynamics MSE"),
        ("dyn_rel", "Relative dynamics error (down)"),
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
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

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
    import matplotlib.pyplot as plt

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
