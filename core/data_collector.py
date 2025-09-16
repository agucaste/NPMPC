import os
import pickle

import numpy as np
from constructor import constructor

from typing import Tuple
from numpy.typing import NDArray
from tqdm import tqdm

from nn_policy import NNRegressor
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

def run_trajectory(x0: np.ndarray, steps: int, simulator: Simulator, controller: NNRegressor | MPC) -> Tuple[NDArray, NDArray]:        
    """
    Runs a closed loop trajectory from an initial condition x0, using the specified 'controller'.

    Args:
        simulator: the do-mpc simulator.
        controller: either an MPC controller or an NNRegressor policy.
        x0: initial condition, shape (n, 1) or (n, ).
    Returns:
        x_hist: history of states, shape (steps, n).
        u_hist: history of controls, shape (steps, m).
    """
    simulator.x0 = x0
    if isinstance(controller, MPC):
        controller.x0 = x0
        controller.set_initial_guess()

    x_hist = []
    u_hist = []
    x = x0
    for t in range(steps):
        u = controller.make_step(x)
        x_hist.append(x)
        u_hist.append(u)
        x = simulator.make_step(u)
    return np.array(x_hist).squeeze(), np.array(u_hist).squeeze()

class DataCollector:
    """
    Class that, given a model, an mpc controller and a simulator,
        - Generates trajectories,
        - Saves trajectories to memory."""
    def __init__(self, model, mpc, simulator):
        self.model = model
        self.mpc = mpc
        self.simulator = simulator

        self.steps = 1000  # problem horizon.
        assert mpc.settings.n_horizon <= self.steps, "MPC horizon must be less than or equal to simulation steps."
        
        self.data = {
            'x': [],  # states
            'dx': [], # derivatives
            'u': [],  # control
        }
        assert mpc.settings.n_horizon <= self.steps

    def collect_data(self, num_trajectories: int, lb: np.ndarray | float, ub: np.ndarray | float, method: str = 'random') -> dict[str, list]:
        """
        Collect data by running trajectories from random initial conditions.
        Args:
            num_trajectories: number of trajectories to collect.
            lb: lower bound for initial condition.
            ub: upper bound for initial condition.
            method: 'random' or 'grid' for sampling initial conditions.
        """
        # Dimension of state.
        n = self.model.x.shape[0]
        # Cast bounds to arrays if needed.
        lb = lb * np.ones(n) if isinstance(lb, (float, int)) else lb
        ub = ub * np.ones(n) if isinstance(ub, (float, int)) else ub

        assert method in ['random', 'grid']    
        if method == 'random':
            # Uniform sampling over [lb, ub]
            X0 = np.random.uniform(lb, ub, size=(num_trajectories, n))            
        elif method == 'grid':
            # Creates a uniform grid of initial conditions.
            axs = [np.linspace(lb[i], ub[i], int(np.ceil(num_trajectories**(1/len(lb))))) for i in range(len(lb))]
            grid = np.meshgrid(*axs)
            X0 = np.c_[tuple(g.ravel() for g in grid)]
                        
        for t in tqdm(range(num_trajectories), desc="Collecting trajectories"):
            x0 = X0[t].reshape(-1,1)
            x, u = run_trajectory(x0=x0, steps=self.steps, simulator=self.simulator, controller=self.mpc)
            self.data['x'].append(x)
            self.data['u'].append(u)

        return self.get_data()
    
    def get_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        # This method shouldn't be used.
        raise DeprecationWarning        
        return self.model._rhs_fun(x, u, [], [], [], [])

    def get_data(self):
        return self.data
    
    def save_data(self, path: str, filename: str) -> None:
        """
        Saves the dataset to a file.
        Args:
            - path: directory path to save the file.
            - filename: name of the file.
        """
        full_path = os.path.join(path, filename + '.pkl')
        with open(full_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {full_path}")


if __name__ == "__main__":
    env = 'min_time'
    model, mpc, simulator = constructor('min_time')
    collector = DataCollector(model, mpc, simulator)
    data = collector.collect_data(num_trajectories=3**2, lb=-2, ub=2, method='grid')
    
    print(f"Collected data keys: {list(data.keys())}")
    for key, value in data.items():
        print(f"{key}: shape {np.array(value).shape}")

    x = np.array(data['x'])
    
    import matplotlib.pyplot as plt

    # x has shape (T, steps, state_dim)
    # Plot each row (trajectory) in the same plot
    for traj in x:
        plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=5, alpha=0.5)  

    plt.xlabel(r'$p$')
    plt.ylabel(r'$q$')
    plt.title('Trajectories (MPC)')
    # plt.legend()
    plt.savefig("example_trajectories.png", dpi=300)
    plt.show()