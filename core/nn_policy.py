"""
Author: Agustin Castellano (@agucaste)
"""

import os

import numpy as np
import faiss as fa



from typing import Tuple
from numpy.typing import NDArray


def configure_faiss_threads(threads: int | None = None):
    """Limit FAISS OpenMP parallelism so concurrent runs do not saturate the machine."""
    threads = int(os.environ.get("MINT_FAISS_THREADS", "2")) if threads is None else int(threads)
    if threads > 0:
        fa.omp_set_num_threads(threads)
    return threads


configure_faiss_threads()


class NNRegressor(object):
    x: fa.IndexFlat  # the state data
    u: NDArray # the controls
    nx: int  # state dimension
    nu: int  # control dimension

    def __init__(self, nx: int, nu: int, k: int = 1):
        self.x = fa.IndexFlatL2(nx)  # the state data, shape (N, nx)
        self.u = np.empty((0, nu), dtype=float)

        self.nx = nx  # state dimension
        self.nu = nu  # control dimension

        self._k: int = k  # number of neighbors
    
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
    
    @property
    def name(self) -> str:
        return f'MINT_{self.size}'

    def add_data(self, x: list | np.ndarray, u: list | np.ndarray, drop_xn: bool = False):
        """
        Adds data to the nearest neighbor regressor.
        Args:
            x: list of states, each of shape (T+1, n), T is the episode length, n is state dimension.
                note this includes the terminal state, which is throwed away.
            u: list of controls, each of shape (T, m) or (T, ).
            drop_xn: whether to drop the final state (if it comes from MPC)
        """
        if isinstance(x, list):                      
            x = np.array([xi[:-1] for xi in x]) if drop_xn else np.array([xi[:] for xi in x])  # (num_trajectories, T, n)
            x = np.concatenate(x, axis=0)
        else:
            if drop_xn:
                x = x[:-1]
        if isinstance(u, list):
            u = np.concatenate(u, axis=0)
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1, self.nx))
        u = np.asarray(u, dtype=float).reshape(-1, self.nu)
        # Throw away last point
        # print(f"Adding data of size x: {x.shape}, u: {u.shape}")
        assert x.shape[0] == u.shape[0]
        # print(f"x shape: {x.shape}, u shape: {u.shape}")
        self.x.add(x)  # type: ignore         
        self.u = np.concatenate([self.u, u], axis=0)
        NNRegressor._check_consistency(self)

        # print(f'how many us? {len(self.u)}')
        # print(f'how many x? {self.x.ntotal}')  # type: ignore

    def set_data(self, x: list | np.ndarray, u: list | np.ndarray, drop_xn: bool = False):
        """
        Replaces the regressor dataset while keeping this object alive.
        """
        self.x.reset()
        self.u = np.empty((0, self.nu), dtype=float)
        self.add_data(x, u, drop_xn=drop_xn)

    def remove_data(self, indices: list | np.ndarray):
        """
        Removes rows from the FAISS index and the aligned control array.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return

        indices = np.ascontiguousarray(np.unique(indices))
        assert np.all(indices >= 0), "Cannot remove negative indices."
        assert np.all(indices < self.size), f"Cannot remove indices >= policy size {self.size}."

        selector = fa.IDSelectorBatch(indices.size, fa.swig_ptr(indices))
        removed = self.x.remove_ids(selector)  # type: ignore
        assert removed == indices.size, f"Expected to remove {indices.size} rows, removed {removed}."

        self.u = np.delete(self.u, indices, axis=0)
        NNRegressor._check_consistency(self)

    def _check_consistency(self):
        assert self.x.ntotal == self.u.shape[0], (
            f"FAISS index has {self.x.ntotal} points, but u has {self.u.shape[0]}"
        )
    
    def query(self, xq: np.ndarray) -> Tuple[NDArray, NDArray]:
        """
        Query the nearest neighbor.
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
        Returns:
            i: index corresponding to the nearest neighbor, shape (k, )
        """
        k = min(self._k, self.size)
        xq = np.ascontiguousarray(np.asarray(xq, dtype=np.float32).reshape(-1, self.nx))
        D, I = self.x.search(xq, k)  # type: ignore # actual search
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
        D = D.reshape(-1)
        I = I.reshape(-1)
        if self._k == 1:
            return self.u[I[0]].reshape(1, -1)
        else:
            if w == 'distance':
                D_inv = 1 / (D + 1e-6)  # avoid division by zero            
                return (self.u[I].T @ D_inv / np.sum(D_inv)).reshape(1, -1)
            elif w == 'equal':
                return np.mean(self.u[I], axis=0).reshape(1, -1)
            else:
                raise ValueError(f"Unknown weighting method: '{w}'")


class MINTPolicy(NNRegressor):
    """
    A nearest neighbor policy class that extends NNRegressor.
    The main difference is:
        - It adds (xi, ui, Ji, ri) to the dataset (state, control, cost-to-go, feasibility radius).
        - When picking actions, it is greedy w.r.t. upper bound of cost-to-go.
            - This is achieved by:
                1. performing similarity search to get the top neighbors,
                2. picking the one that minimizes upper bound.
    """

    J: NDArray # the optimal cost-to-go

    def __init__(self, nx: int, nu: int, k: int = 1, lambd: float = 1.0):
        self.lambd = lambd  # weight for distance in cost-to-go upper bound
        super().__init__(nx, nu, k)
        self.J = np.empty((0,), dtype=float)  # cost-to-go
    
    def add_data(self, x: list | np.ndarray, u: list | np.ndarray, J: list | np.ndarray):
        """
        Adds data to the nearest neighbor regressor.
        Args:
            x: list of states, each of shape (T+1, n), T is the episode length, n is state dimension.
                note this includes the terminal state, which is throwed away.
            u: list of controls, each of shape (T, m) or (T, ).
        """
        old_size = self.size
        super().add_data(x, u)

        J = np.asarray(J, dtype=float).reshape(-1)
        added = self.size - old_size
        assert J.shape[0] == added, f"Expected {added} new J values, got {J.shape[0]}"

        self.J = np.concatenate([self.J, J], axis=0)
        self._check_consistency()
        
        # print(f"size of x: {self.x.ntotal}\nsize of u: {len(self.u)}\nsize of J: {len(self.J)}")
        # raise Exception

    def set_data(self, x: list | np.ndarray, u: list | np.ndarray, J: list | np.ndarray):
        """
        Replaces the policy dataset while keeping this object alive.
        """
        self.x.reset()
        self.u = np.empty((0, self.nu), dtype=float)
        self.J = np.empty((0,), dtype=float)
        self.add_data(x, u, J)

    def update_values(self, J: list | np.ndarray):
        """
        Replaces Q/J values for the current dataset without touching FAISS.
        """
        J = np.asarray(J, dtype=float).reshape(-1)
        assert J.shape[0] == self.size, f"Expected {self.size} J values, got {J.shape[0]}"
        self.J = J
        self._check_consistency()

    def remove_data(self, indices: list | np.ndarray):
        """
        Removes rows from the FAISS index, controls, and aligned values.
        """
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            return

        indices = np.ascontiguousarray(np.unique(indices))
        assert np.all(indices >= 0), "Cannot remove negative indices."
        assert np.all(indices < self.size), f"Cannot remove indices >= policy size {self.size}."

        selector = fa.IDSelectorBatch(indices.size, fa.swig_ptr(indices))
        removed = self.x.remove_ids(selector)  # type: ignore
        assert removed == indices.size, f"Expected to remove {indices.size} rows, removed {removed}."

        self.u = np.delete(self.u, indices, axis=0)
        self.J = np.delete(self.J, indices, axis=0)
        self._check_consistency()

    def _check_consistency(self):
        super()._check_consistency()
        assert self.x.ntotal == self.J.shape[0], (
            f"FAISS index has {self.x.ntotal} points, but J has {self.J.shape[0]}"
        )
        
    def make_step(self, xq: np.ndarray, w: str = 'equal') -> np.ndarray:
        """
        Decides an action based on the state. 
        The policy is greedy w.r.t. upper bound of cost-to-go:
            u(x) = u_i, where i = argmin_{i in k-NN(x)} Ji + lambda * ||x - xi||
        Args:
            xq: query states, shape (N, n) or (n, ), N is number of queries, n is state dimension.
            w: weighting method, 'equal' or 'distance'.
        Returns:
            u: controls corresponding to the nearest neighbors, shape (N, m) or (m, ) if k=1.
        """ 
        D, I = self.query(xq)       
        D, I = D.reshape(-1), I.reshape(-1)    

        # print(f"Query: {xq}, distances: {D}, indices: {I}")
        if self._k == 1:
            return self.u[I[0]].reshape(1, -1)
        else:
            # print(f"Indices of neighbors: {I}, shape: {I.shape},\ndistances: {D}, shape: {D.shape}")
            # print(f"J has length {len(self.J)}")
            D = np.sqrt(np.maximum(D, 0.0))  # FAISS IndexFlatL2 returns squared L2 distances.
            J_ub = self.J[I] + self.lambd * D  # Build upper bound
            i = np.argmin(J_ub)  # Act greedily
            return self.u[I[i]].reshape(1, -1)
    
    @property
    def name(self) -> str:
        return f'MINT_{self.size}_k{self.k}_l{self.lambd}'


class EncodedMINTPolicy(MINTPolicy):
    """MINT policy whose FAISS index lives in frozen encoder space."""

    def __init__(self, encoder, nx: int, nu: int, k: int = 1, lambd: float = 1.0):
        self.encoder = encoder
        super().__init__(nx=nx, nu=nu, k=k, lambd=lambd)

    def make_step(self, xq: np.ndarray, w: str = 'equal') -> np.ndarray:
        zq = self.encoder.encode(xq)
        return super().make_step(zq, w=w)

    @property
    def name(self) -> str:
        return f'Encoded{super().name}_{self.encoder.name}'

        



    


if __name__ == "__main__":
    # Import DataCollector here to avoid circular import when module is imported
    from data_collector import DataCollector, run_trajectory
    from constructor import constructor
    from config import get_default_kwargs_yaml

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
