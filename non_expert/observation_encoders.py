from __future__ import annotations

import hashlib
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class ObservationEncoder(Protocol):
    name: str
    input_dim: int | None
    output_dim: int | None

    def encode(self, obs: np.ndarray) -> np.ndarray:
        ...


@dataclass
class IdentityEncoder:
    input_dim: int
    name: str = "identity"

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        return np.ascontiguousarray(obs, dtype=np.float32)


@dataclass
class SklearnPCAEncoder:
    scaler: object
    pca: object
    path: Path
    metadata: dict
    name: str = "pca"

    @property
    def input_dim(self) -> int | None:
        return int(self.scaler.n_features_in_) if hasattr(self.scaler, "n_features_in_") else None

    @property
    def output_dim(self) -> int:
        if hasattr(self.pca, "n_components_"):
            return int(self.pca.n_components_)
        return int(self.pca.n_components)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        z = self.pca.transform(self.scaler.transform(obs))
        return np.ascontiguousarray(z, dtype=np.float32)


def encode_trajectories(trajectories, encoder: ObservationEncoder):
    encoded = []
    for X, U, C, X_next in trajectories:
        encoded.append((
            encoder.encode(X),
            U,
            C,
            encoder.encode(X_next),
        ))
    return encoded


def env_id_to_encoder_key(env_id: str) -> str:
    return env_id
    """Convert Gym ids like InvertedPendulum-v5 to InvertedPendulum."""
    # return re.sub(r"-v\d+$", "", env_id)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pca_encoder(path: Path) -> SklearnPCAEncoder:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if "scaler" not in payload or "pca" not in payload:
        raise ValueError(f"Expected scaler and pca in encoder pickle: {path}")

    metadata = {k: v for k, v in payload.items() if k not in {"scaler", "pca"}}
    metadata["path"] = str(path)
    metadata["sha256"] = sha256_file(path)
    return SklearnPCAEncoder(
        scaler=payload["scaler"],
        pca=payload["pca"],
        path=path,
        metadata=metadata,
    )


def default_encoder_path(project_root: Path, env_key: str) -> Path:
    return project_root / "non_expert" / "minari" / "outputs" / env_key / "pca_encoder.pkl"


def load_observation_encoder(config, env_id: str, obs_dim: int, project_root: Path) -> ObservationEncoder:
    encoder_cfg = getattr(config, "encoder", None)
    mode = getattr(encoder_cfg, "mode", "auto") if encoder_cfg is not None else "auto"
    path = getattr(encoder_cfg, "path", None) if encoder_cfg is not None else None

    print(f"Loading observation encoder with mode={mode} from path={path} for env_id={env_id} and obs_dim={obs_dim}")

    if mode == "none":
        return IdentityEncoder(input_dim=obs_dim)

    if mode == "path":
        if not path:
            raise ValueError("encoder.mode='path' requires encoder.path")
        encoder_path = Path(path).expanduser()
        if not encoder_path.is_absolute():
            encoder_path = project_root / encoder_path
    elif mode == "auto":
        env_key = env_id_to_encoder_key(env_id)
        encoder_path = default_encoder_path(project_root, env_key)
        print(f"Auto encoder path for env_id {env_id} (key {env_key}): {encoder_path}")
        if not encoder_path.exists():
            print(f"No encoder found at {encoder_path}, using identity encoder instead.")
            return IdentityEncoder(input_dim=obs_dim)
    else:
        raise ValueError(f"Unknown encoder mode: {mode}")

    encoder = load_pca_encoder(encoder_path)
    if encoder.input_dim is not None and encoder.input_dim != obs_dim:
        raise ValueError(
            f"Encoder input dim {encoder.input_dim} does not match observation dim {obs_dim}: {encoder_path}"
        )
    return encoder
