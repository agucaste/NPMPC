import numpy as np
from time import time
from matplotlib import pyplot as plt

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from nn_policy import NNRegressor
from config import Config
from data_collector import run_trajectory

from typing import Any, Callable
from numpy.typing import NDArray
import warnings


class Sampler:
    def __init__(self, X0: NDArray | None = None, f: Callable | None = None):
        # The sampling index
        assert (X0 is None) ^ (f is None), "Provide exactly one of X0 or f"
        self.i = 0

        if X0 is not None:
            # Will sample from a grid, indexed by rows in X0.
            self.X0 = X0
            self.n = X0.shape[0]
            self._sampler = self._sample_from_grid
        else:
            self.f = f
            self._sampler = self._sample_from_function

    def _sample_from_grid(self):
        x = self.X0[self.i]
        self.i = (self.i + 1) % self.n
        if self.i == 0:
            warnings.warn("All elements have been sampled once, repeating samples now...")
        return x

    def _sample_from_function(self):
        return self.f() # type: ignore

    def sample(self):
        return self._sampler()
    

class Evaluator():
    def __init__(self) -> None:
        self.stats: dict[str, Any] = {}
        pass

    def evaluate(self,
                 model: Model,
                 controller: MPC | NNRegressor,
                 simulator: Simulator,
                 cfgs: Config,
                 sampler: Sampler,
                 M: int = 100,) -> None:
        
        
        key = 'MPC' if isinstance(controller, MPC) else f'NN_{controller._k}_D{controller.size}'        
        self.stats[key] = {'t': [], 'x': [], 'u': []}        
        if key != 'MPC':
            # Get the size of the dataset
            self.stats[key].update({'size': controller.size})
        

        steps = cfgs.N        
        for _ in trange(M, desc=f"Evaluating {key}"):
            x0 = sampler.sample()
            start_time = time()
            x, u = run_trajectory(x0, steps, simulator, controller)
            t = time() - start_time

            self.stats[key]['t'].append(t)
            self.stats[key]['x'].append(x)
            self.stats[key]['u'].append(u)
    
    def plot_times(self, filename: str = 'times.pdf') -> None:
        """
        Makes a histogram of the times taken for each controller.
        """
        plt.style.use('bmh')
        plt.figure()
        plt.xscale('log')
        for key, stat in self.stats.items():
            times = np.array(stat['t'])
            bins = np.logspace(np.log10(times.min()), np.log10(times.max()), int(np.sqrt(len(times))))
            m = np.mean(times)
            label = key + f' (mean={m:.4f}s)'
            plt.hist(times, bins=bins, alpha=0.6, edgecolor='black', label=label)

        plt.xlabel('Time (s)')
        plt.ylabel('Occurrences')
        plt.legend()
        title = "Time taken to run one trajectory"
        plt.title(title)
        plt.savefig(filename)
        plt.show()

if __name__ == "__main__":
    from constructor import constructor
    from data_collector import DataCollector
    from config import get_default_kwargs_yaml
    from tqdm import trange

    # Define the system and data collector
    env = 'pendulum'    
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    model, mpc, simulator = constructor(env)
    collector = DataCollector(model, mpc, simulator)
        
    G = [3, 5, 7, 9, 11]
    regressors = [NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0]) for _ in G]
    for nn, g in zip(regressors, G):
        # Collect data uniformly.
        data = collector.collect_data(num_trajectories=g**2, lb=-2, ub=2, method='grid')
        nn.add_data(data['x'], data['u'])

    


    M = 100  # Number of trajectories to evaluate
    samplers = {'X0': np.random.uniform(-3, 3, size=(M, nn.nx, 1)),}
                # 'f': lambda: np.random.uniform(-3, 3, size=(nn.nx, 1))}
    for k, v in samplers.items():
        print(f"Testing sampler with {k}")
        sampler = Sampler(**{k: v})
        evaluator = Evaluator()
        # Evaluate MPC controller
        evaluator.evaluate(model, mpc, simulator, cfgs, sampler, M)        
        # Evaluate NN controller for different k
        for nn in regressors:
            # nn.set_k(k)
            evaluator.evaluate(model, nn, simulator, cfgs, sampler, M)
        # Plot results.
        evaluator.plot_times(filename=f'times_d_{nn.size}_M_{M}.pdf')
    



