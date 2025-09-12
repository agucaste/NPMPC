import numpy as np
from numpy.typing import NDArray
import faiss as fa

from constructor import constructor
from data_collector import DataCollector

class NNRegressor(object):
    x: fa.IndexFlat  # the state data
    u: list[NDArray] # the controls
    
    def __init__(self, nx: int, nu: int):
        self.x = fa.IndexFlatL2(nx)  # the state data, shape (N, nx)
        self.u = []

        self.nx = nx  # state dimension
        self.nu = nu

    def add_data(self, x: list | np.ndarray, u: list | np.ndarray):
        """
        Adds data to the nearest neighbor regressor.
        Args:
            x: list of states, each of shape (T, n), T is the episode length, n is state dimension.
            u: list of controls, each of shape (T, m) or (T, ).
        """
        if isinstance(x, list):
            x = np.concatenate(x, axis=0)
        if isinstance(u, list):
            u = np.concatenate(u, axis=0)
        assert x.shape[0] == u.shape[0]
        print(f"x shape: {x.shape}, u shape: {u.shape}")
        self.x.add(x)  # type: ignore         
        for c in u:
            self.u.append(c.reshape(1, -1))
            #TODO: Can i make this work well, maybe without concat.

        print(f'how many us? {len(self.u)}')
        print(f'how many x? {self.x.ntotal}')  # type: ignore
    
    def query(self, xq: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Query the nearest neighbor.
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
            k: number of nearest neighbors to return.
        Returns:
            i: index corresponding to the nearest neighbor, shape (k, )
        """
        D, I = self.x.search(xq.T, k)  # type: ignore # actual search
        return I
        
    def act(self, xq: np.ndarray, k: int = 1) -> np.ndarray:
        """
        Act method for compatibility with policy interface.
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
            k: number of nearest neighbors to return.
        Returns:
            u: controls corresponding to the nearest neighbors, shape (N, m) or (m, ) if k=1.
        """        
        I = self.query(xq, k=k).squeeze()        
        if k == 1:
            return self.u[I]
        else:
            return np.mean(np.array([self.u[j] for j in I]), axis=0)

        return self.query(xq, k)
    


if __name__ == "__main__":
    model, mpc, simulator = constructor('pendulum')
    collector = DataCollector(model, mpc, simulator)
    data = collector.collect_data(num_trajectories=3**2, lb=-2, ub=2, method='grid')

    nn = NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0])
    nn.add_data(data['x'], data['u'])

    x0 = np.random.normal(loc=0.0, scale=1.0, size=model.x.shape)

    for k in [1,3,5]:
        print(f"Control action for k={k}: {nn.act(x0, k=k)}, of shape {nn.act(x0, k=k).shape}")


    def run_trajectory(self, x0: np.ndarray):
        self.simulator.x0 = x0

        x_hist = []
        dx_hist = []
        u_hist = []

        x = x0
        for k in range(collector.steps):
            u = nn.act(x)
            # dx = self.get_velocity(x, u)

            x_hist.append(x)
            # dx_hist.append(dx)
            u_hist.append(u)

            x = self.simulator.make_step(u)
        return np.array(x_hist).squeeze()


    trajectories = []
    num_trajectories = 10
    lb, ub = -3, 3
    X0 = np.random.uniform(lb, ub, size=(num_trajectories, nn.nx, 1))            
    for x0 in X0:            
        traj = run_trajectory(collector, x0)
        trajectories.append(traj)
    
    import matplotlib.pyplot as plt

    # x has shape (T, steps, state_dim)
    # Plot each row (trajectory) in the same plot
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=5, alpha=0.5)  
    for traj in collector.data['x']:
        if traj is collector.data['x'][0]:
            plt.plot(traj[:, 0], traj[:, 1], 'k', marker='x', linewidth=.5, alpha=0.3, label='Demonstrations')
        else:
            plt.plot(traj[:, 0], traj[:, 1], 'k', marker='x', linewidth=.5, alpha=0.3)
    plt.xlabel(r'$p$')
    plt.ylabel(r'$q$')
    plt.title('Trajectories (NN)')
    plt.legend()
    plt.savefig("example_trajectories_nn.png", dpi=300)
    plt.show()

