import numpy as np
from time import time
from matplotlib import pyplot as plt

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from nn_policy import NNRegressor
from config import Config
from data_collector import run_trajectory, get_trajectory_cost

from typing import Any, Callable, Optional
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
    def __init__(self, mpc: MPC) -> None:
        self.stats: dict[str, Any] = {}
        self.mpc: MPC = mpc
        self.steps = None  # type: ignore
        pass

    def evaluate(self,
                 model: Model,
                 controller: Optional[NNRegressor],
                 simulator: Simulator,
                 cfgs: Config,
                 sampler: Sampler,
                 M: int = 100,) -> None:
        
        if controller is None:
            controller = self.mpc
        if isinstance(controller, MPC):
            key = 'MPC_' + str(controller.settings.n_horizon)
        else:
            key = f'NN_{controller._k}_D{controller.size}'        
        self.stats[key] = {'t': [], 'x': [], 'u': [], 'c': []}        
        if not key.startswith('MPC'):
            # Get the size of the dataset
            self.stats[key].update({'size': controller.size})
        
        self.steps = cfgs.N 
        for _ in trange(M, desc=f"Evaluating {key}"):
            x0 = sampler.sample()
            start_time = time()
            x, u = run_trajectory(x0, self.steps, simulator, controller)
            t = time() - start_time
            c = get_trajectory_cost(mpc_t, x, u)

            # print(f'added cost is {c}')

            self.stats[key]['t'].append(t)
            self.stats[key]['x'].append(x)
            self.stats[key]['u'].append(u)
            self.stats[key]['c'].append(c)

    
    def plot_histograms(self, filename: str = 'times.pdf') -> None:
        """
        Makes a histogram of the times taken for each controller.
        """
        plt.style.use('bmh')
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].set_xscale('log')
        axs[1].set_xscale('log')        
        for i, (key, stat) in enumerate(self.stats.items()):
            # Plotting trajectory times
            times = np.array(stat['t'])            
            bins = np.logspace(np.log10(times.min()), np.log10(times.max()), int(np.sqrt(len(times))))
            m = np.mean(times)
            label = key + f' (mean={m:.4f}s)'
            axs[0].hist(times, bins=bins, alpha=0.6, edgecolor='black', label=label)
            axs[0].set_yscale('log')
            # axs[0].set_xticks([1])
            # axs[0].set_xticklabels([label])

            # Plotting trajectory costs
            costs = np.array(stat['c'])
            # print(f'costs is: {costs}')
            bins = np.logspace(np.log10(costs.min()), np.log10(costs.max()), int(np.sqrt(len(costs))))
            m = np.mean(costs)
            label = key + f' (mean={m:.4f}s)'
            axs[1].hist(costs, bins=bins, alpha=0.6, edgecolor='black', label=label)

        for ax in axs:
            ax.legend()
            ax.set_ylabel('Occurrences')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_title("Time taken to run one trajectory")
        axs[1].set_title("Cost-to-go ")
        axs[1].set_xlabel("Cost")
        plt.savefig(filename)
        # plt.show()

    def plot_boxplots(self, filename: str = 'times.pdf') -> None:
        """
        Makes a histogram of the times taken for each controller.
        """
        # plt.style.use('bmh')
        fig, axs = plt.subplots(1, 2, figsize=(16, 5))
        
        n = len(self.stats)
                
        T = self.steps
        mpc_costs = self.stats[f'MPC_{T}']['c']
        for k, v in self.stats.items():
            if k == f'MPC_{T}':
                v['gap'] = []
            else:
                # v['gap'] = [(J - mpc_costs[i]) for i, J in enumerate(v['c']) ]
                v['gap'] = [(J - mpc_costs[i])/(mpc_costs[i] + 1e-6) for i, J in enumerate(v['c']) ]
        print(f"MPC gaps are:")
        print(self.stats[f'MPC_{T}']['gap'])
        what_to_plot = ['t', 'gap']
        for i, ax in enumerate(axs):
        
            bp = ax.boxplot([self.stats[key][what_to_plot[i]] for key in self.stats.keys()],
                            patch_artist=True,
                            positions=np.arange(n), widths=0.6,
                            boxprops=dict(color='k'),
                            medianprops=dict(color='k', linewidth=2),
                            meanprops=dict(marker="^", markersize=6, markeredgecolor='green',  markerfacecolor='green'),
                            showmeans=True)
            
            # assign colors one by one
            for patch, color in zip(bp['boxes'], ['C' + str(i) for i in range(n)]):
                patch.set_facecolor(color)        
                patch.set_alpha(0.6)
            ax.set_xticks(np.arange(n))
            ax.set_xticklabels(self.stats.keys())
            ax.set_yscale('log')
            if i == 0:                
                ax.set_ylabel('Time (s)')
                ax.set_title("Time taken to run one trajectory")
            else:
                ax.set_ylabel('Gap')
                ax.set_title(r"Empirical normalized gap $\frac{J^{\pi}-J^{\star}}{J^\star}$")
                ax.set_ylim(bottom=1e-8)
        M = len(self.stats[f'MPC_{T}']['t'])
        plt.suptitle(f"Statistics over M={M} trajectories, horizon T={T}")
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()        
    
    def plot_tradeoff(self, filename: str = 'tradeoff.pdf') -> None:
        """
        Plots the trade-off between online computation times (x-axis)
        and cost-to-go (y-axis) for each controller family (MPC and NN)"""
        T = self.steps
        assert 'gap' in self.stats[f'MPC_{T}'], "Gaps not computed yet, run plot_boxplots first."
        plt.style.use('bmh')

        keys = list(self.stats.keys())
        nn_controllers = [k for k in keys if not k.startswith('MPC')]
        mpc_controllers = [k for k in keys if k.startswith('MPC')]

        nn_stats = []
        mpc_stats = []

        self.stats[f'MPC_{T}']['gap'] = [0]
        for key in keys:
            t = np.median(self.stats[key]['t'])
            gap = np.median(self.stats[key]['gap'])
            print(f"Controller {key}: median time {t:.4f}s, median gap {gap:.4f}")
            if key.startswith('MPC'):
                mpc_stats.append((t, gap))
            else:
                nn_stats.append((t, gap))

        plt.figure(figsize=(8,6))
        plt.plot(*zip(*mpc_stats), 's-', c='C0', label='MPC', markersize=16, linewidth=2)
        plt.plot(*zip(*nn_stats), '*-', c='C1', label='NPP (Ours)', markersize=16, linewidth=2)
        plt.xscale('log')
        plt.xlabel("Online Computation Time (s)")
        plt.ylabel("Normalized Gap")
        plt.title("Trade-off between computation time and cost-to-go")
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        






    
    def plot_trajectories(self, filename: str = 'trajectories.pdf') -> None:
        """
        Plots all the trajectories for each controller.
        """
        plt.style.use('bmh')
        plt.figure()
        for key, stat in self.stats.items():
            for traj in stat['x']:
                plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=5, alpha=0.5, label=key if traj is stat['x'][0] else None)  
        
        plt.xlabel(r'$p$')
        plt.ylabel(r'$q$')
        plt.title(f'Trajectories for different controllers')
        plt.legend()
        plt.savefig(filename)
        # plt.show()

if __name__ == "__main__":
    from constructor import constructor
    from data_collector import DataCollector
    from config import get_default_kwargs_yaml
    from tqdm import trange
    import os

    # Define the system and data collector
    env = 'min_time'    
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    model, mpc_t, mpc_h, simulator = constructor(env, cfgs)
    collector = DataCollector(model, mpc_t, simulator, cfgs)
        
    G = [5, 7, 9, 11]  # [3, 5, 7, 9, 11]  # grid anchors per dimension
    regressors = [NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0]) for _ in G]
    
    ub = 2  # upper bound for grid
    method = 'grid'
    for nn, g in zip(regressors, G):
        # Collect data uniformly.
        data = collector.collect_data(num_trajectories=g**2, lb=-ub, ub=ub, method=method)
        nn.add_data(data['x'], data['u'])

        filename = f"{env}_{method}{g}_T{cfgs.N}_H{cfgs.mpc.n_horizon}_D{nn.size}.pkl"
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(path, 'datasets')
        collector.save_data(path=path, filename=filename)

        

    


    M = 100  # Number of trajectories to evaluate
    samplers = {'X0': np.random.uniform(-3, 3, size=(M, nn.nx, 1)),}
                # 'f': lambda: np.random.uniform(-3, 3, size=(nn.nx, 1))}
    for k, v in samplers.items():
        print(f"Testing sampler with {k}")
        sampler = Sampler(**{k: v})
        evaluator = Evaluator(mpc_t)
        # Evaluate full horizon MPC controller
        evaluator.evaluate(model, None, simulator, cfgs, sampler, M)        
        # Evaluate receding horizon MPC controller
        evaluator.evaluate(model, mpc_h, simulator, cfgs, sampler, M)        
        # Evaluate NN controller for different k
        for nn in regressors:
            # nn.set_k(k)
            evaluator.evaluate(model, nn, simulator, cfgs, sampler, M)
        # Plot results.
        # evaluator.plot_histograms(filename=f'times_{env}_d_{nn.size}_M_{M}.pdf')
        evaluator.plot_boxplots(filename=f'bp_{env}_d_{nn.size}_M_{M}.pdf')
        evaluator.plot_tradeoff(filename=f'tradeoff_{env}_d_{nn.size}_M_{M}.pdf')




