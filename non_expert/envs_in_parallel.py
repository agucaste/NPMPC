"""Utilities for running MuJoCo environments from the same state in parallel."""

import gymnasium as gym
import numpy as np


class MuJoCoStateWrapper(gym.Wrapper):
    """Expose full-state get/set helpers for MuJoCo vector environments."""

    def __init__(self, env):
        """Initialize the wrapper and cache MuJoCo state dimensions."""
        super().__init__(env)
        if not hasattr(self.unwrapped, "model") or not hasattr(self.unwrapped, "data"):
            raise TypeError("MuJoCoStateWrapper requires a MuJoCo environment.")
        self.nq = self.unwrapped.model.nq
        self.nv = self.unwrapped.model.nv

    def get_mujoco_state(self):
        """Return the concatenated qpos/qvel MuJoCo state."""
        return np.concatenate([
            self.unwrapped.data.qpos.copy(),
            self.unwrapped.data.qvel.copy(),
        ])

    def set_mujoco_state(self, state):
        """Set the environment to a concatenated qpos/qvel MuJoCo state."""
        state = np.asarray(state, dtype=float).reshape(-1)
        expected_dim = self.nq + self.nv
        if state.shape[0] != expected_dim:
            raise ValueError(f"Expected state dim {expected_dim}, got {state.shape[0]}")

        qpos = state[:self.nq]
        qvel = state[self.nq:self.nq + self.nv]
        self.unwrapped.set_state(qpos, qvel)
        return self.unwrapped._get_obs()

    def step(self, action):
        """Clip the action before stepping so vector rollouts match MujocoSystem."""
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)


def make_env(env_id, seed):
    """Build a thunk for one seeded MuJoCo environment worker."""
    def thunk():
        """Create one wrapped MuJoCo environment instance."""
        env = gym.make(env_id)
        env = MuJoCoStateWrapper(env)
        env.reset(seed=seed)
        return env

    return thunk


def make_vine_envs(env_id: str, num_envs: int, seed: int):
    """Create persistent asynchronous MuJoCo envs for vine trajectory collection."""
    if int(num_envs) < 1:
        raise ValueError(f"num_envs must be at least one, got {num_envs}")

    return gym.vector.AsyncVectorEnv(
        [make_env(env_id, seed=int(seed) + i) for i in range(int(num_envs))]
    )


def reset_vine_envs_same_state(envs, seed=None):
    """Reset vector envs and synchronize all workers to worker zero's state."""
    if seed is None:
        envs.reset()
    else:
        envs.reset(seed=[int(seed) + i for i in range(envs.num_envs)])

    states = envs.call("get_mujoco_state")
    obs = envs.call("set_mujoco_state", states[0])
    return np.stack(obs), states[0]


# -------------------------
# Create vectorized env
# -------------------------
def main():
    """Run a small manual smoke test for synchronized vector MuJoCo envs."""
    N = 4
    env_id = "Swimmer-v5"

    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_id, seed=1000 + i) for i in range(N)]
    )

    obs, info = envs.reset()

    print("Initial obs shape:", obs.shape)


    # -------------------------
    # Save one reference state
    # -------------------------
    # For a real use case, you may get this from some previous trajectory.

    single_env = gym.make(env_id)
    single_env.reset(seed=0)

    qpos_ref = single_env.unwrapped.data.qpos.copy()
    qvel_ref = single_env.unwrapped.data.qvel.copy()

    single_env.close()


    # -------------------------
    # Reset all envs to same state
    # -------------------------
    print(f"Setting:\nq = {qpos_ref}\nqdot = {qvel_ref}")
    state_ref = np.concatenate([qpos_ref, qvel_ref])
    obs_tuple = envs.call("set_mujoco_state", state_ref)
    obs = np.stack(obs_tuple)

    print("Obs shape after manual state reset:", obs.shape)
    print(f"Obs after manual reset:")
    for o in obs:
        print(o)


    # -------------------------
    # Step all envs at once
    # -------------------------

    actions = np.stack([
        envs.single_action_space.sample()
        for _ in range(N)
    ])

    obs, rewards, terminated, truncated, infos = envs.step(actions)

    print("Obs shape after step:", obs.shape)
    print("Rewards shape:", rewards.shape)
    print("Terminated shape:", terminated.shape)
    print("Truncated shape:", truncated.shape)


    envs.close()

if __name__ == "__main__":
    main()
