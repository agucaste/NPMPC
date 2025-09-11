from constructor import constructor

import numpy as np

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
        
        self.data = {
            'x': [],  # states
            'dx': [], # derivatives
            'u': [],  # control
        }
        assert mpc.settings.n_horizon <= self.steps

    def run_trajectory(self, x0: np.ndarray):
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

            x_hist.append(x)
            dx_hist.append(dx)
            u_hist.append(u)

            x = self.simulator.make_step(u)
        
        self.data['x'].append(np.array(x_hist).squeeze())
        self.data['dx'].append(np.array(dx_hist).squeeze())
        self.data['u'].append(np.array(u_hist).squeeze())

    def collect_data(self, num_trajectories: int, lb: np.ndarray | float, ub: np.ndarray | float, method: str = 'random'):
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
                        
        for t in range(num_trajectories):
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



    data = collector.collect_data(num_trajectories=7**2, lb=-2, ub=2, method='grid')
    print(data['x'])
    

    # X0 = np.meshgrid(
    #     np.linspace(-2, 2, 5),
    #     np.linspace(-2, 2, 5)
    # )
    # X0 = np.c_[tuple(X.ravel() for X in X0)]
    # print(f"X0 shape: {X0.shape}")
    # T = X0.shape[0]
    # T = 10
    # for t in range(T):
    #     print(f' collecting trajectory {t+1}/{T}')
    #     x0 = X0[t].reshape(-1,1)
    #     print(f"x0 = {x0}")
    #     collector.run_trajectory(x0)
    
    # data = collector.get_data()

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