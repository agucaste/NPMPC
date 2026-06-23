"""Regenerate encoder diagnostic plots from a saved dynamics encoder."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from non_expert.minari.utils import (
    DATASET,
    FrozenEncoder,
    encode_normalized_observations,
    flatten_transitions,
    load_minari_datasets,
    neighborhood_preservation_overlaps,
    plot_latent_scatter,
    plot_latent_spectrum,
    plot_neighborhood_preservation,
    plot_training_history,
)


def load_saved_encoder(path: Path) -> tuple[FrozenEncoder, dict, list[dict[str, float]]]:
    """Load a saved dynamics encoder pickle into a frozen encoder payload.

    Args:
        path: Path to a ``dynamics_encoder.pkl`` artifact created by
            ``non_expert.minari.encoder``.

    Returns:
        The frozen encoder, saved diagnostics, and training history.

    Raises:
        ValueError: If the pickle is not a dynamics encoder artifact.
    """
    with path.open("rb") as f:
        payload = pickle.load(f)

    if payload.get("type") != "dynamics":
        raise ValueError(f"Expected a dynamics encoder artifact, got {payload.get('type')!r}.")

    encoder = FrozenEncoder(
        obs_scaler=payload["obs_scaler"],
        act_scaler=payload["act_scaler"],
        model_state_dict=payload["model_state_dict"],
        obs_dim=payload["obs_dim"],
        act_dim=payload["act_dim"],
        latent_dim=payload["latent_dim"],
        hidden=payload["hidden"],
    )
    return encoder, payload.get("diagnostics", {}), payload.get("history", [])


def load_neighbor_values(config: dict, override: list[int] | None) -> list[int]:
    """Return neighborhood sizes from CLI overrides or saved configuration.

    Args:
        config: Saved encoder run configuration.
        override: Optional CLI-provided neighborhood sizes.

    Returns:
        Neighborhood sizes to evaluate.
    """
    if override is not None:
        return override
    return list(config.get("neighbor_values", [100, 200, 500, 1000]))


def main() -> None:
    """Regenerate diagnostic plots for a saved encoder artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("encoder_path", type=Path)
    parser.add_argument("--env", choices=DATASET.keys(), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-plot-points", type=int, default=None)
    parser.add_argument("--max-neighborhood-points", type=int, default=None)
    parser.add_argument("--neighbor-values", type=int, nargs="+", default=None)
    parser.add_argument("--filename-prefix", default=None)
    args = parser.parse_args()

    encoder_path = args.encoder_path.resolve()
    config_path = encoder_path.parent / "config.json"
    config = {}
    if config_path.exists():
        with config_path.open("r") as f:
            config = json.load(f)

    env = args.env or config.get("env")
    if env is None:
        raise ValueError("Pass --env or keep the original config.json next to the encoder pickle.")

    output_dir = args.output_dir or encoder_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or int(config.get("batch_size", 1024))
    seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    max_plot_points = args.max_plot_points or int(config.get("max_plot_points", 20_000))
    max_neighborhood_points = args.max_neighborhood_points or int(
        config.get("max_neighborhood_points", 10_000)
    )
    neighbor_values = load_neighbor_values(config, args.neighbor_values)
    filename_prefix = args.filename_prefix
    if filename_prefix is None:
        filename_prefix = "best_" if encoder_path.name == "dynamics_encoder_best.pkl" else ""

    encoder, diagnostics, history = load_saved_encoder(encoder_path)
    dataset_ids = DATASET[env]

    print(f"Using saved encoder: {encoder_path}")
    print(f"Using datasets: {dataset_ids}")

    datasets = load_minari_datasets(dataset_ids, force_download=args.force_download)
    transitions = flatten_transitions(datasets)

    model = encoder.build_model().to(args.device)
    z = encode_normalized_observations(
        model=model,
        obs=transitions.obs,
        obs_scaler=encoder.obs_scaler,
        device=args.device,
        batch_size=batch_size,
    )
    obs_n = encoder.obs_scaler.transform(transitions.obs).astype(np.float32)
    overlaps_by_k = neighborhood_preservation_overlaps(
        obs_n=obs_n,
        z=z,
        max_points=max_neighborhood_points,
        neighbor_values=neighbor_values,
        seed=seed,
    )

    if history:
        plot_training_history(history, output_dir, filename_prefix=filename_prefix)
    plot_latent_scatter(
        z=z,
        labels=transitions.dataset_labels,
        episode_returns=transitions.episode_returns,
        time_in_episode=transitions.time_in_episode,
        dataset_ids=dataset_ids,
        output_dir=output_dir,
        max_points=max_plot_points,
        seed=seed,
        filename_prefix=filename_prefix,
    )
    plot_latent_spectrum(z, output_dir, filename_prefix=filename_prefix)
    plot_neighborhood_preservation(
        overlaps_by_k=overlaps_by_k,
        output_dir=output_dir,
        obs_n=obs_n,
        z=z,
        max_points=max_neighborhood_points,
        neighbor_values=neighbor_values,
        seed=seed,
        filename_prefix=filename_prefix,
    )

    for n_neighbors, overlaps in overlaps_by_k.items():
        diagnostics[f"{n_neighbors}nn_preservation"] = float(np.mean(overlaps))

    print("Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.6g}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
