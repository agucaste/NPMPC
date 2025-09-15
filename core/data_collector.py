import numpy as np
from constructor import constructor

from typing import Tuple
from numpy.typing import NDArray
from tqdm import tqdm

class DataCollector:
    """
    Class that, given a model, an mpc controller and a simulator,
        - Generates trajectories,
        - Saves trajectories to memory."""
    def __init__(self, model, mpc, simulator):
        self.model = model
        self.mpc = mpc
        self.simulator = simulator

        self.steps = 100  # problem horizon.
        assert mpc.settings.n_horizon <= self.steps, "MPC horizon must be less than or equal to simulation steps."
        
        self.data = {
            'x': [],  # states
            'dx': [], # derivatives
            'u': [],  # control
        }
        assert mpc.settings.n_horizon <= self.steps

    def run_trajectory(self, x0: np.ndarray) -> Tuple[NDArray, NDArray, NDArray]:
        self.simulator.x0 = x0
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()

        x_hist = []
        dx_hist = []
        u_hist = []

        x = x0
        for k in range(self.steps):
            u = self.mpc.make_step(x)
            dx = self.get_velocity(x, u)
            # Track x, u and dx
            x_hist.append(x)
            dx_hist.append(dx)
            u_hist.append(u)
            # Simulate
            x = self.simulator.make_step(u)

            print(self.simulator.data['_x'])
            print(f' has length {len(self.mpc.data["_x"])}')
            raise ValueError
        
        # Convert to arrays
        x_hist = np.array(x_hist).squeeze()
        dx_hist = np.array(dx_hist).squeeze()
        u_hist = np.array(u_hist).squeeze()
        # Store in memory
        self.data['x'].append(x_hist)
        self.data['dx'].append(dx_hist)
        self.data['u'].append(u_hist)

        return x_hist, u_hist, dx_hist

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
            self.run_trajectory(x0)

        return self.get_data()
    
    def get_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.model._rhs_fun(x, u, [], [], [], [])
        self.data['reference'].append(reference)

    def get_data(self):
        return self.data
    
if __name__ == "__main__":
    model, mpc, simulator = constructor('pendulum')
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