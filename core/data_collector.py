import os
import pickle
from time import time

import numpy as np
from constructor import constructor
from config import Config, get_default_kwargs_yaml

from typing import Tuple
from numpy.typing import NDArray
from tqdm import tqdm

from nn_policy import NNRegressor
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
from do_mpc.model import Model

def get_stage_cost(mpc: MPC) -> float:    
    raise DeprecationWarning
    x_t = mpc.data['_x'][-1]  # State at time step 0
    u_t = mpc.data['_u'][-1]  # Control input at time step 0
    z_t = mpc.data['_z'][-1]  # Algebraic state at time step 0 (if applicable)
    tvp_t = mpc.data['_tvp'][-1]  # Time-varying parameters at time step 0
    p_t = mpc.data['_p'][-1]  # Parameters at time step 0
    return mpc.lterm_fun(x_t, u_t, z_t, tvp_t, p_t)

def get_terminal_cost(mpc: MPC) -> float:
    raise DeprecationWarning
    x_t = mpc.data['_x'][-1]  # State at time step 0
    z_t = mpc.data['_z'][-1]  # Algebraic state at time step 0 (if applicable)
    tvp_t = mpc.data['_tvp'][-1]  # Time-varying parameters at time step 0
    p_t = mpc.data['_p'][-1]  # Parameters at time step 0
    return mpc.mterm_fun(x_t, z_t, tvp_t, p_t)

def get_trajectory_cost(mpc: MPC, x: NDArray, u: NDArray) -> float:
    """
    Computes the cost of a trajectory of (x, u) pairs.
    MPC is used to access cost functions, but the trajectory may come from MPC or NNRegressor.
    Args:
        mpc: the MPC controller with cost functions.
        x: states
        u: controls
    """
    assert x.shape[0] == u.shape[0] + 1, "x should have one more time step than u"
    
    cost = 0.    
    for t in range(u.shape[0]):
        cost += mpc.lterm_fun(x[t], u[t], [], [], [])
    cost += mpc.mterm_fun(x[-1], [], [])
    return float(cost)

def count_infeasible_steps(x: NDArray, cfgs: Config, mode: str = 'l1') -> int:
    """
    Counts the number of time steps in the trajectory (x, u) that violate state constraints.
    Args:
        x: states
        cfgs: config object
    Returns:
        L_1 norm of constraint violations, summed over the trajectory.
    """
    assert mode in ['counts', 'l1']
    if not (hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub')):
        return 0
    count = 0
    norm = 0.0
    for t in range(x.shape[0]):
        if np.any(x[t] < cfgs.x_lb) or np.any(x[t] > cfgs.x_ub):
            # Integer counts
            count += 1
            # L_1 norm of violations
            lb_count = np.maximum(cfgs.x_lb - x[t], 0)
            ub_count = np.maximum(x[t] - cfgs.x_ub, 0)
            norm += lb_count.sum(where=lb_count>0) + ub_count.sum(where=ub_count>0)
    
    return count if mode == 'counts' else norm
    


def run_trajectory(x0: np.ndarray, steps: int, simulator: Simulator, controller: NNRegressor | MPC) -> Tuple[NDArray, NDArray, NDArray]:        
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
    t_hist = []
    x = x0
    for t in range(steps):
        start_time = time()
        u = controller.make_step(x)
        t = time() - start_time
        # Shouldn't use full-length trajectory in one go because of collocation errors!
        # u_seq = np.asarray(controller.opt_x_num['_u', :, 0])  # (steps, m, 1)        
        # x_seq = np.asarray(controller.opt_x_num['_x', :, 0]) # (steps, n, c, 1) where c may be the number of collocation points        
        x_hist.append(x)
        u_hist.append(u)
        t_hist.append(t)
        x = simulator.make_step(u)
    x_hist.append(x)
    
    x_hist = np.array(x_hist)  # (steps+1, n, 1)
    u_hist = np.array(u_hist)  # (steps, m, 1)
    t_hist = np.array(t_hist) 
    return x_hist.squeeze(), u_hist.squeeze(), t_hist.squeeze()

class DataCollector:
    """
    Class that, given a model, an mpc controller and a simulator,
        - Generates trajectories,
        - Saves trajectories to memory.
    """
    def __init__(self, model: Model, mpc: MPC, simulator: Simulator, cfgs: Config):
        self.model = model
        self.mpc = mpc
        self.simulator = simulator
        self.cfgs = cfgs

        self.f_name: str | None = None  # filename to save the dataset.

        self.steps = cfgs.N  # problem horizon.
        assert mpc.settings.n_horizon <= self.steps, "MPC horizon must be less than or equal to simulation steps."
        
        self.data = {
            'x': [],  # states
            'dx': [], # derivatives
            'u': [],  # control
            't': [],  # time taken to compute control
            'c': 0.0,   # cost,
            'i': 0     # number of steps that violate constraints
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
            x, u, t = run_trajectory(x0=x0, steps=self.steps, simulator=self.simulator, controller=self.mpc)
            c = get_trajectory_cost(self.mpc, x, u)
            i = count_infeasible_steps(x, self.cfgs)
            self.data['x'].append(x)
            self.data['u'].append(u)
            self.data['t'].append(t)
            self.data['c'] = c  # TODO: Make it a list of costs per trajectory.
            self.data['i'] = i

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
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {full_path}")


if __name__ == "__main__":
    env = 'pendulum'
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    print(f"Configs are {cfgs}")
    
    model, mpc, simulator = constructor(env, cfgs)
    collector = DataCollector(model, mpc, simulator, cfgs)
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
    plt.title(f'Trajectories for {env} (MPC)')
    # plt.legend()
    plt.savefig(f"example_trajectories_{env}.png", dpi=300)
    plt.show()