"""
Script to test parallel environments in MuJoCo. The goal is to be able to reset all environments to the same initial state
"""

import gymnasium as gym
import numpy as np


class MuJoCoStateWrapper(gym.Wrapper):
    """
    Wrapper that allows to set the state of a mujoco env manually, and exposes the observation.
    """
    def set_mujoco_state(self, qpos, qvel):
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)

        self.unwrapped.set_state(qpos, qvel)

        # Return observation after manually setting state
        return self.unwrapped._get_obs()


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = MuJoCoStateWrapper(env)
        env.reset(seed=seed)
        return env

    return thunk


# -------------------------
# Create vectorized env
# -------------------------
def main():
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
    obs_tuple = envs.call("set_mujoco_state", qpos_ref, qvel_ref)
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