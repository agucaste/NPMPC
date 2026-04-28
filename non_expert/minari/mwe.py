"""
Minimal working example of using minari to:
    - Download offline datasets,
    - Train PCA / Encoder
    - Visualize the results.
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

# Keep matplotlib/fontconfig from trying to write caches under the home
# directory, which is not writable in some project/sandbox setups.
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import minari
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


DATASET = {
    "InvertedPendulum": (
        "mujoco/invertedpendulum/medium-v0",
        "mujoco/invertedpendulum/expert-v0",
    ),
    "HalfCheetah": (
        "mujoco/halfcheetah/medium-v0",
        "mujoco/halfcheetah/expert-v0",
    ),
}
ENV = "InvertedPendulum"


@dataclass
class PCAEncoder:
    scaler: StandardScaler
    pca: PCA

    def encode(self, o: np.ndarray) -> np.ndarray:
        """Encode one observation or a batch of observations."""
        o = np.asarray(o)
        is_single_observation = o.ndim == 1
        if is_single_observation:
            o = o[None, :]

        z = self.pca.transform(self.scaler.transform(o))
        return z[0] if is_single_observation else z

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Map a latent vector or batch back into observation space."""
        z = np.asarray(z)
        is_single_latent = z.ndim == 1
        if is_single_latent:
            z = z[None, :]

        obs_norm = self.pca.inverse_transform(z)
        obs = self.scaler.inverse_transform(obs_norm)
        return obs[0] if is_single_latent else obs


def download_and_load_datasets(dataset_ids: tuple[str, ...], force_download: bool = False):
    datasets = {}
    for dataset_id in dataset_ids:
        if force_download:
            print(f"Downloading {dataset_id}...")
            minari.download_dataset(dataset_id, force_download=True)
            datasets[dataset_id] = minari.load_dataset(dataset_id)
        else:
            print(f"Loading {dataset_id} (downloads if missing)...")
            datasets[dataset_id] = minari.load_dataset(dataset_id, download=True)
    return datasets


def flatten_observations(dataset, dataset_id: str) -> dict[str, np.ndarray]:
    observations = []
    labels = []
    episode_returns = []
    time_in_episode = []

    for episode in dataset.iterate_episodes():
        obs = np.asarray(episode.observations)
        if obs.ndim != 2:
            raise ValueError(f"Expected 2D observations, got shape {obs.shape}")

        n = obs.shape[0]
        observations.append(obs)
        labels.extend([dataset_id] * n)
        episode_returns.extend([float(np.sum(episode.rewards))] * n)
        time_in_episode.extend(np.linspace(0.0, 1.0, n))

    return {
        "observations": np.concatenate(observations, axis=0),
        "labels": np.asarray(labels),
        "episode_returns": np.asarray(episode_returns),
        "time_in_episode": np.asarray(time_in_episode),
    }


def fit_pca_encoder(obs: np.ndarray, requested_components: int) -> PCAEncoder:
    n_components = min(requested_components, obs.shape[1])
    if n_components != requested_components:
        print(
            f"Requested {requested_components} PCA components, but observations are "
            f"{obs.shape[1]}D; using n_components={n_components}."
        )

    scaler = StandardScaler()
    obs_norm = scaler.fit_transform(obs)

    pca = PCA(n_components=n_components)
    pca.fit(obs_norm)
    return PCAEncoder(scaler=scaler, pca=pca)


def sample_indices(n: int, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def plot_explained_variance(encoder: PCAEncoder, output_dir: Path) -> None:
    explained = encoder.pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    xs = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(xs, explained, label="Individual")
    ax.plot(xs, cumulative, marker="o", color="black", label="Cumulative")
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("How much observation variance the encoder keeps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "explained_variance.pdf")
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
    if z.shape[1] < 2:
        print("Skipping 2D latent scatter because the encoder has only one component.")
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
        short_label = dataset_id.split("/")[-1]
        axs[0].scatter(
            z_sample[mask, 0],
            z_sample[mask, 1],
            s=7,
            alpha=0.35,
            label=short_label,
        )
    axs[0].set_title("Dataset")
    axs[0].legend(markerscale=2)

    positive_returns = returns_sample[returns_sample > 0]
    norm = None
    if positive_returns.size > 0:
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
    axs[1].set_title("Episode return (log color)")
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
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    fig.suptitle("Latent space diagnostics")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_scatter.pdf")
    plt.close(fig)


def plot_reconstruction_quality(
    encoder: PCAEncoder,
    obs: np.ndarray,
    output_dir: Path,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    idx = sample_indices(obs.shape[0], max_points=max_points, rng=rng)
    obs_sample = obs[idx]
    z_sample = encoder.encode(obs_sample)
    obs_hat = encoder.reconstruct(z_sample)
    errors = np.linalg.norm(obs_sample - obs_hat, axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].hist(errors, bins=60, color="#4477aa", alpha=0.85)
    axs[0].set_title("Reconstruction error")
    axs[0].set_xlabel("||observation - reconstruction||2")
    axs[0].set_ylabel("count")

    axs[1].scatter(obs_sample.ravel(), obs_hat.ravel(), s=4, alpha=0.25)
    lower = min(float(obs_sample.min()), float(obs_hat.min()))
    upper = max(float(obs_sample.max()), float(obs_hat.max()))
    axs[1].plot([lower, upper], [lower, upper], color="black", linewidth=1)
    axs[1].set_title("Original vs reconstructed coordinates")
    axs[1].set_xlabel("original")
    axs[1].set_ylabel("reconstructed")

    fig.tight_layout()
    fig.savefig(output_dir / "reconstruction_quality.pdf")
    plt.close(fig)

    print(
        "Reconstruction error: "
        f"mean={errors.mean():.4g}, median={np.median(errors):.4g}, "
        f"95th percentile={np.percentile(errors, 95):.4g}"
    )


def neighborhood_preservation_score(
    obs: np.ndarray,
    z: np.ndarray,
    max_points: int,
    n_neighbors: int,
    seed: int,
) -> float:
    """Mean fraction of original nearest neighbors preserved in latent space."""
    rng = np.random.default_rng(seed)
    idx = sample_indices(obs.shape[0], max_points=max_points, rng=rng)
    obs_sample = obs[idx]
    z_sample = z[idx]

    obs_dist = pairwise_distances(obs_sample)
    z_dist = pairwise_distances(z_sample)

    # Drop self-neighbor at column 0.
    obs_nn = np.argsort(obs_dist, axis=1)[:, 1 : n_neighbors + 1]
    z_nn = np.argsort(z_dist, axis=1)[:, 1 : n_neighbors + 1]

    overlaps = [
        len(set(obs_neighbors).intersection(z_neighbors)) / n_neighbors
        for obs_neighbors, z_neighbors in zip(obs_nn, z_nn)
    ]
    return float(np.mean(overlaps))


def save_encoder(
    encoder: PCAEncoder,
    env_name: str,
    dataset_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    path = output_dir / "pca_encoder.pkl"
    payload = {
        "env": env_name,
        "dataset_ids": dataset_ids,
        "scaler": encoder.scaler,
        "pca": encoder.pca,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=DATASET.keys(), default=ENV)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--max-neighborhood-points", type=int, default=2_000)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir / args.env
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_ids = DATASET[args.env]
    print(f"Using {args.env}: {dataset_ids}")

    datasets = download_and_load_datasets(dataset_ids, force_download=args.force_download)
    flattened = [flatten_observations(datasets[dataset_id], dataset_id) for dataset_id in dataset_ids]

    obs = np.concatenate([x["observations"] for x in flattened], axis=0)
    labels = np.concatenate([x["labels"] for x in flattened], axis=0)
    episode_returns = np.concatenate([x["episode_returns"] for x in flattened], axis=0)
    time_in_episode = np.concatenate([x["time_in_episode"] for x in flattened], axis=0)

    print(f"Fitting PCA encoder on {obs.shape[0]} observations with obs_dim={obs.shape[1]}.")
    encoder = fit_pca_encoder(obs, requested_components=args.components)
    z = encoder.encode(obs)

    print("Explained variance ratio:", np.round(encoder.pca.explained_variance_ratio_, 4))
    print("Cumulative explained variance:", np.round(np.cumsum(encoder.pca.explained_variance_ratio_), 4))

    score = neighborhood_preservation_score(
        obs,
        z,
        max_points=args.max_neighborhood_points,
        n_neighbors=args.neighbors,
        seed=args.seed,
    )
    print(f"{args.neighbors}-NN preservation in latent space: {score:.3f}")

    encoder_path = save_encoder(encoder, args.env, dataset_ids, output_dir)

    plot_explained_variance(encoder, output_dir)
    plot_latent_scatter(
        z,
        labels,
        episode_returns,
        time_in_episode,
        dataset_ids,
        output_dir,
        max_points=args.max_plot_points,
        seed=args.seed,
    )
    plot_reconstruction_quality(
        encoder,
        obs,
        output_dir,
        max_points=args.max_plot_points,
        seed=args.seed,
    )

    print(f"Saved encoder to {encoder_path}")
    print(f"Saved visualizations to {output_dir}")


if __name__ == "__main__":
    main()
