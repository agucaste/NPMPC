"""
A base wrapper for MuJoCo environments.

Date: 04/17/2026
Author: Agustin Castellano (@agucaste)

"""

import numpy as np


class MujocoSystem:
    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        self.nq = self.base_env.model.nq
        self.nv = self.base_env.model.nv
        self.state_dim = self.nq + self.nv
        self.nx = self.env.observation_space.shape[0]
        self.nu = self.env.action_space.shape[0]
        self.u_lb = self.env.action_space.low
        self.u_ub = self.env.action_space.high

    def reset(self, seed=None):
        obs, _ = self.env.reset(seed=seed)
        return obs

    def set_state(self, s):
        s = np.asarray(s, dtype=float).reshape(-1)
        assert s.shape[0] == self.state_dim, f"Expected state dim {self.state_dim}, got {s.shape[0]}"

        qpos = s[:self.nq]
        qvel = s[self.nq:self.nq + self.nv]
        self.base_env.set_state(qpos, qvel)

    def get_state(self):
        return np.concatenate([
            self.base_env.data.qpos.copy(),
            self.base_env.data.qvel.copy(),
        ])

    def clip_action(self, u):
        return np.clip(np.asarray(u, dtype=float).reshape(-1), self.u_lb, self.u_ub)

    def transition(self, u):
        obs_next, reward, terminated, truncated, info = self.env.step(self.clip_action(u))
        return -reward, obs_next, terminated or truncated


if __name__ == "__main__":
    import os

    import gymnasium as gym

    from core.config import Config, load_yaml

    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, "mujoco_config.yaml")
    config = Config.dict2config(load_yaml(cfg_path)["defaults"])

    env_ids = [
        config.env_id,
    ]

    for env_id in env_ids:
        env = gym.make(env_id)
        system = MujocoSystem(env)

        obs = system.reset(seed=0)
        s0 = system.get_state()
        obs = system.reset(seed=10)
        # print(f"current observation is {obs}, of shape {obs.shape}")
        # print(f"current full state is {s0}, of shape {s0.shape}")
        u0 = env.action_space.sample()

        system.set_state(s0)
        assert np.allclose(system.get_state(), s0), "State after setting does not match the original state."
        # print(f"Full state after setting is {system.get_state()}")
        cost, obs_next, done = system.transition(u0)

        print(
            f"{env_id}: "
            f"obs_dim={system.nx}, state_dim={system.state_dim}, action_dim={system.nu}, "
            f"cost={cost:.3f}, done={done}, next_obs_shape={obs_next.shape}"
        )

        env.close()
