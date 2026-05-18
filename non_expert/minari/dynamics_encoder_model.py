from __future__ import annotations

import torch
import torch.nn as nn


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
        z_next_pred = self.dynamics_head(za)
        r_pred = self.reward_head(za).squeeze(-1)
        return z, z_next_pred, r_pred
