from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


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


@dataclass
class TorchDynamicsEncoder:
    obs_scaler: object
    model_state_dict: dict
    obs_dim: int
    act_dim: int
    latent_dim: int
    hidden: int
    path: Path
    metadata: dict
    name: str = "dynamics"
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the Torch model once so encode calls are cheap."""
        from non_expert.minari.utils import DynamicsEncoder

        self._model = DynamicsEncoder(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
        )
        self._model.load_state_dict(self.model_state_dict)
        self._model.eval()

    @property
    def input_dim(self) -> int:
        """Return the raw observation dimension expected by the encoder."""
        return int(self.obs_dim)

    @property
    def output_dim(self) -> int:
        """Return the latent dimension produced by the encoder."""
        return int(self.latent_dim)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode one observation or a batch into frozen dynamics latents."""
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        obs_n = self.obs_scaler.transform(obs).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs_n, dtype=torch.float32)
            z = self._model.encode(x).cpu().numpy()
        return np.ascontiguousarray(z, dtype=np.float32)


@dataclass
class TorchAutoEncoder:
    obs_scaler: object
    model_state_dict: dict
    obs_dim: int
    latent_dim: int
    hidden: int
    path: Path
    metadata: dict
    name: str = "autoencoder"
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the Torch model once so encode calls are cheap."""
        from non_expert.minari.autoencoder import AutoEncoder

        self._model = AutoEncoder(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden=self.hidden,
        )
        self._model.load_state_dict(self.model_state_dict)
        self._model.eval()

    @property
    def input_dim(self) -> int:
        """Return the raw observation dimension expected by the encoder."""
        return int(self.obs_dim)

    @property
    def output_dim(self) -> int:
        """Return the latent dimension produced by the encoder."""
        return int(self.latent_dim)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode one observation or a batch into frozen autoencoder latents."""
        obs = np.asarray(obs, dtype=np.float32)
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        obs_n = self.obs_scaler.transform(obs).astype(np.float32)
        with torch.no_grad():
            x = torch.as_tensor(obs_n, dtype=torch.float32)
            z = self._model.encode(x).cpu().numpy()
        return np.ascontiguousarray(z, dtype=np.float32)


def encode_trajectories(trajectories, encoder: ObservationEncoder):
    """Encode trajectory observations as float32 features.

    Args:
        trajectories: Raw trajectories as ``(obs, actions, costs, next_obs)``
            tuples.
        encoder: Observation encoder that maps raw observations to features.

    Returns:
        Encoded trajectories as ``(Z, U, C, Z_next)`` tuples. ``Z`` and
        ``Z_next`` are contiguous float32 arrays; controls and costs are left
        unchanged.
    """
    encoded = []
    for X, U, C, X_next in trajectories:
        encoded.append((
            np.ascontiguousarray(encoder.encode(X), dtype=np.float32),
            U,
            C,
            np.ascontiguousarray(encoder.encode(X_next), dtype=np.float32),
        ))
    return encoded


def env_id_to_encoder_key(env_id: str) -> str:
    """Return the encoder artifact key for a Gym environment id."""
    return env_id


def sha256_file(path: Path) -> str:
    """Compute a stable content hash for an artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_metadata(path: Path, payload: dict, excluded_keys: set[str]) -> dict:
    """Build common metadata for loaded encoder artifacts."""
    metadata = {k: v for k, v in payload.items() if k not in excluded_keys}
    metadata["path"] = str(path)
    metadata["sha256"] = sha256_file(path)
    return metadata


def _load_pickle(path: Path) -> dict:
    """Load an encoder artifact pickle."""
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected encoder artifact to contain a dict: {path}")
    return payload


def _build_pca_encoder(path: Path, payload: dict) -> SklearnPCAEncoder:
    """Build a PCA encoder from a loaded artifact payload."""
    if "scaler" not in payload or "pca" not in payload:
        raise ValueError(f"Expected scaler and pca in encoder pickle: {path}")

    return SklearnPCAEncoder(
        scaler=payload["scaler"],
        pca=payload["pca"],
        path=path,
        metadata=_artifact_metadata(path, payload, {"scaler", "pca"}),
    )


def _build_dynamics_encoder(path: Path, payload: dict) -> TorchDynamicsEncoder:
    """Build a Torch dynamics encoder from a loaded artifact payload."""
    required = [
        "obs_scaler",
        "model_state_dict",
        "obs_dim",
        "act_dim",
        "latent_dim",
        "hidden",
    ]
    if "encoder" in payload:
        frozen_encoder = payload["encoder"]
        if any(not hasattr(frozen_encoder, name) for name in required):
            raise ValueError(f"Expected frozen dynamics encoder in artifact: {path}")
        kwargs = {name: getattr(frozen_encoder, name) for name in required}
        excluded_keys = {"encoder", "history"}
    else:
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"Missing dynamics encoder fields {missing} in artifact: {path}")
        kwargs = {name: payload[name] for name in required}
        excluded_keys = set(required) | {"act_scaler", "history"}

    return TorchDynamicsEncoder(
        **kwargs,
        path=path,
        metadata=_artifact_metadata(path, payload, excluded_keys),
    )


def _build_autoencoder(path: Path, payload: dict) -> TorchAutoEncoder:
    """Build a Torch autoencoder from a loaded artifact payload."""
    required = [
        "obs_scaler",
        "model_state_dict",
        "obs_dim",
        "latent_dim",
        "hidden",
    ]
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Missing autoencoder fields {missing} in artifact: {path}")

    return TorchAutoEncoder(
        **{name: payload[name] for name in required},
        path=path,
        metadata=_artifact_metadata(path, payload, set(required) | {"history"}),
    )


def load_pca_encoder(path: Path) -> SklearnPCAEncoder:
    """Load a PCA observation encoder artifact."""
    return _build_pca_encoder(path, _load_pickle(path))


def load_encoder(path: Path) -> ObservationEncoder:
    """Load any supported observation encoder artifact."""
    payload = _load_pickle(path)
    if "scaler" in payload and "pca" in payload:
        return _build_pca_encoder(path, payload)
    if payload.get("type") == "autoencoder":
        return _build_autoencoder(path, payload)
    if "encoder" in payload or "model_state_dict" in payload:
        return _build_dynamics_encoder(path, payload)
    raise ValueError(f"Unsupported encoder artifact format: {path}")


def default_encoder_path(project_root: Path, env_key: str) -> Path:
    """Return the default auto-discovered PCA encoder path for an environment."""
    return project_root / "non_expert" / "minari" / "outputs" / env_key / "pca" / "pca_encoder.pkl"


def load_observation_encoder(config, env_id: str, obs_dim: int, project_root: Path) -> ObservationEncoder:
    """Load the configured observation encoder for a MuJoCo experiment."""
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

    encoder = load_encoder(encoder_path)
    if encoder.input_dim is not None and encoder.input_dim != obs_dim:
        raise ValueError(
            f"Encoder input dim {encoder.input_dim} does not match observation dim {obs_dim}: {encoder_path}"
        )
    return encoder
