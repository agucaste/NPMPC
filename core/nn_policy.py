import numpy as np
import faiss as fa

from constructor import constructor
from config import get_default_kwargs_yaml

from typing import Tuple
from numpy.typing import NDArray

class NNRegressor(object):
    x: fa.IndexFlat  # the state data
    u: list[NDArray] # the controls
    nx: int  # state dimension
    nu: int  # control dimension

    def __init__(self, nx: int, nu: int):
        self.x = fa.IndexFlatL2(nx)  # the state data, shape (N, nx)
        self.u = []

        self.nx = nx  # state dimension
        self.nu = nu  # control dimension

        self._k: int = 1  # number of neighbors
    
    def set_k(self, k: int):
        """
        Sets the number of neighbors.
        """
        assert isinstance(k, int) and k >= 1, "k must be a positive integer."
        self._k = k
        return
    
    @property
    def k(self) -> int:
        return self._k
    
    @property
    def size(self) -> int:
        return self.x.ntotal  # type: ignore

    def add_data(self, x: list | np.ndarray, u: list | np.ndarray):
        """
        Adds data to the nearest neighbor regressor.
        Args:
            x: list of states, each of shape (T+1, n), T is the episode length, n is state dimension.
                note this includes the terminal state, which is throwed away.
            u: list of controls, each of shape (T, m) or (T, ).
        """
        if isinstance(x, list):          
            x = np.array([xi[:-1] for xi in x])  # (num_trajectories, T, n)
            x = np.concatenate(x, axis=0)
        else:
            x = x[:-1]
        if isinstance(u, list):
            u = np.concatenate(u, axis=0)
        # Throw away last point
        print(f"Adding data of size x: {x.shape}, u: {u.shape}")
        assert x.shape[0] == u.shape[0]
        print(f"x shape: {x.shape}, u shape: {u.shape}")
        self.x.add(x)  # type: ignore         
        for c in u:
            self.u.append(c.reshape(1, -1))
            #TODO: Can i make this work well, maybe without concat.

        print(f'how many us? {len(self.u)}')
        print(f'how many x? {self.x.ntotal}')  # type: ignore
    
    def query(self, xq: np.ndarray) -> Tuple[NDArray, NDArray]:
        """
        Query the nearest neighbor.
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
        Returns:
            i: index corresponding to the nearest neighbor, shape (k, )
        """
        D, I = self.x.search(xq.T, self._k)  # type: ignore # actual search
        return D, I
        
    def make_step(self, xq: np.ndarray, w: str = 'equal') -> np.ndarray:
        """
        Decides an action based on the state. For k=1, returns the control of the nearest neighbor.
        For k>1, returns a combination of the k-nearest controls, based on the weighting method
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
            w: weighting method, 'equal' or 'distance'.
        Returns:
            u: controls corresponding to the nearest neighbors, shape (N, m) or (m, ) if k=1.
        """ 
        D, I = self.query(xq)       
        I = I.squeeze()        
        if self._k == 1:
            return self.u[I]
        else:
            if w == 'distance':
                D = D.squeeze()
                D_inv = 1 / (D + 1e-6)  # avoid division by zero            
                return np.array([self.u[j] for j in I]) @ D_inv / np.sum(D_inv)
            elif w == 'equal':
                return np.mean(np.array([self.u[j] for j in I]), axis=0)
            else:
                raise ValueError(f"Unknown weighting method: '{w}'")
            
    


if __name__ == "__main__":
    # Import DataCollector here to avoid circular import when module is imported
    from data_collector import DataCollector, run_trajectory

    env = 'min_time'
    # Define the system and data collector
    model, mpc, simulator = constructor(env)
    collector = DataCollector(model, mpc, simulator)
    # Collect data uniformly.
    data = collector.collect_data(num_trajectories=3**2, lb=-2, ub=2, method='grid')
    # Define the policy
    nn = NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0])
    nn.add_data(data['x'], data['u'])

    # x, u, _ = collector.run_trajectory(np.array([[10.0], [-10.0]]))
    # nn.add_data(x, u)

    x0 = np.random.normal(loc=0.0, scale=1.0, size=model.x.shape)

    for k in [1,3,5]:
        nn.set_k(k)
        a = nn.make_step(x0)
        print(f"Control action for k={k}: {a}, of shape {a.shape}")    
    nn.set_k(1)
    
    config = get_default_kwargs_yaml(algo='', env_id=env)

    trajectories = []
    num_trajectories = 10
    lb, ub = -3, 3
    X0 = np.random.uniform(lb, ub, size=(num_trajectories, nn.nx, 1))            
    for x0 in X0:            
        x, u = run_trajectory(x0=x0, steps=config['N'] * 5, simulator=simulator, controller=nn)
        trajectories.append(x)
    
    import matplotlib.pyplot as plt

    # x has shape (T, steps, state_dim)
    # Plot each row (trajectory) in the same plot    
    for traj in collector.data['x']:
        # print(f"traj shape: {traj.shape}")
        # print(f"traj is {traj}")
        if traj is collector.data['x'][0]:
            plt.plot(traj[:, 0], traj[:, 1], 'k', marker='x', linewidth=.5, alpha=0.3, label='Demonstrations')
        else:
            plt.plot(traj[:, 0], traj[:, 1], 'k', marker='x', linewidth=.5, alpha=0.3)
    for traj in trajectories:
        
        plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=5, alpha=0.5, label='NN policy' if traj is trajectories[0] else None)  
    plt.xlabel(r'$p$')
    plt.ylabel(r'$q$')
    plt.title(f'Trajectories for {env} problem')
    plt.legend()
    plt.savefig("example_trajectories_nn.png", dpi=300)
    plt.show()

