"""
Minimal working example of using minari to:
    - Download offline datasets,
    - Train PCA / Encoder
    - Visualize the results.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from non_expert.minari.utils import (
    DATASET,
    flatten_transitions,
    load_minari_datasets,
    neighborhood_preservation_overlaps,
    plot_latent_scatter,
    plot_latent_spectrum,
    sample_indices,
)


@dataclass
class PCAEncoder:
    """Pickle-friendly PCA observation encoder artifact.

    Attributes:
        scaler: Observation normalizer fitted before PCA.
        pca: Fitted PCA model over normalized observations.
    """

    scaler: StandardScaler
    pca: PCA

    def encode(self, o: np.ndarray) -> np.ndarray:
        """Encode one observation or a batch of observations.

        Args:
            o: Observation array with shape ``(obs_dim,)`` or ``(n, obs_dim)``.

        Returns:
            PCA latent array with matching leading batch shape.
        """
        o = np.asarray(o)
        is_single_observation = o.ndim == 1
        if is_single_observation:
            o = o[None, :]

        z = self.pca.transform(self.scaler.transform(o))
        return z[0] if is_single_observation else z

    def reconstruct(self, z: np.ndarray) -> np.ndarray:
        """Map a latent vector or batch back into observation space.

        Args:
            z: PCA latent array with shape ``(latent_dim,)`` or ``(n, latent_dim)``.

        Returns:
            Reconstructed observation array with matching leading batch shape.
        """
        z = np.asarray(z)
        is_single_latent = z.ndim == 1
        if is_single_latent:
            z = z[None, :]

        obs_norm = self.pca.inverse_transform(z)
        obs = self.scaler.inverse_transform(obs_norm)
        return obs[0] if is_single_latent else obs


def fit_pca_encoder(obs: np.ndarray, requested_components: int) -> PCAEncoder:
    """Fit a normalized PCA encoder to flat observations.

    Args:
        obs: Flat observation table with shape ``(n, obs_dim)``.
        requested_components: Requested PCA latent dimensionality.

    Returns:
        Fitted PCA encoder using at most ``obs_dim`` components.
    """
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


def plot_explained_variance(encoder: PCAEncoder, output_dir: Path) -> None:
    """Plot individual and cumulative explained variance ratios.

    Args:
        encoder: Fitted PCA encoder.
        output_dir: Directory where the PDF should be written.
    """
    import matplotlib.pyplot as plt

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


def plot_reconstruction_quality(
    encoder: PCAEncoder,
    obs: np.ndarray,
    output_dir: Path,
    max_points: int,
    seed: int,
) -> None:
    """Plot PCA reconstruction errors and reconstructed coordinate fidelity.

    Args:
        encoder: Fitted PCA encoder.
        obs: Flat observation table used for reconstruction diagnostics.
        output_dir: Directory where the PDF should be written.
        max_points: Maximum number of observations to include.
        seed: Random seed used when subsampling observations.
    """
    import matplotlib.pyplot as plt

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


def save_pca_encoder(
    encoder: PCAEncoder,
    env_name: str,
    dataset_ids: tuple[str, ...],
    output_dir: Path,
) -> Path:
    """Save the fitted PCA encoder and dataset metadata.

    Args:
        encoder: Fitted PCA encoder.
        env_name: Short environment name used to select datasets.
        dataset_ids: Minari dataset ids used for fitting.
        output_dir: Directory where the pickle should be written.

    Returns:
        Path to the saved pickle.
    """
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
    """Train a Minari PCA encoder from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=DATASET.keys(), default="Swimmer")
    parser.add_argument("--components", type=int, default=20)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=20_000)
    parser.add_argument("--max-neighborhood-points", type=int, default=10_000)
    parser.add_argument("--neighbor-values", type=int, nargs="+", default=[100, 200, 500, 1000])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = parser.parse_args()

    output_dir = args.output_dir / f"{args.env}-v5" / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    dataset_ids = DATASET[args.env]
    print(f"Using datasets: {dataset_ids}")

    datasets = load_minari_datasets(dataset_ids, force_download=args.force_download)
    transitions = flatten_transitions(datasets)
    obs = transitions.obs

    print(f"Fitting PCA encoder on {obs.shape[0]} observations with obs_dim={obs.shape[1]}.")
    encoder = fit_pca_encoder(obs, requested_components=args.components)
    z = encoder.encode(obs)
    obs_n = encoder.scaler.transform(obs).astype(np.float32)

    print("Explained variance ratio:", np.round(encoder.pca.explained_variance_ratio_, 4))
    print("Cumulative explained variance:", np.round(np.cumsum(encoder.pca.explained_variance_ratio_), 4))

    overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=obs_n,
        z=z,
        max_points=args.max_neighborhood_points,
        neighbor_values=args.neighbor_values,
        seed=args.seed,
    )
    for n_neighbors, overlaps in overlaps_by_k.items():
        print(f"{n_neighbors}-NN preservation in latent space: {float(np.mean(overlaps)):.3f}")

    encoder_path = save_pca_encoder(encoder, args.env, dataset_ids, output_dir)

    plot_explained_variance(encoder, output_dir)
    plot_latent_spectrum(z, output_dir)
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
