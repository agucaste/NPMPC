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

    def run_simulation(self, x0: np.ndarray):
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


    def get_velocity(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.model._rhs_fun(x, u, [], [], [], [])
        self.data['reference'].append(reference)

    def get_data(self):
        return self.data
    
if __name__ == "__main__":
    model, mpc, simulator = constructor('pendulum')
    collector = DataCollector(model, mpc, simulator)


    X0 = np.meshgrid(
        np.linspace(-2, 2, 4),
        np.linspace(-2, 2, 4)
    )
    X0 = np.c_[tuple(X.ravel() for X in X0)]
    T = X0.shape[0]
    for t in range(T):
        print(f' collecting trajectory {t+1}/{T}')
        x0 = np.random.uniform(low=-2, high=2, size=(2,1))    
        collector.run_simulation(x0)
    
    data = collector.get_data()

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