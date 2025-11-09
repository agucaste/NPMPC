import numpy as np
import time
from matplotlib import pyplot as plt

from tqdm import trange
import os
import pickle

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from nn_policy import NNRegressor, NNPolicy
from config import Config
from data_collector import run_trajectory, get_trajectory_cost, count_infeasible_steps

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
                 controller: NNRegressor | MPC | None,
                 simulator: Simulator,
                 cfgs: Config,
                 sampler: Sampler,
                 M: int = 100,) -> None:
        
        if controller is None:
            controller = self.mpc
        if isinstance(controller, MPC):
            key = 'MPC_' + str(controller.settings.n_horizon)
        else:
            key = controller.name        
        self.stats[key] = {'t': [], 'x': [], 'u': [], 'c': [], 'i': []}        
        if not key.startswith('MPC'):
            # Get the size of the dataset
            self.stats[key].update({'size': controller.size})
        
        self.steps = cfgs.N 
        for _ in trange(M, desc=f"Evaluating {key}"):
            x0 = sampler.sample()
            x, u, t = run_trajectory(x0, self.steps, simulator, controller)
            c = get_trajectory_cost(self.mpc, x, u)
            i = count_infeasible_steps(x, cfgs)

            # print(f'added cost is {c}')

            self.stats[key]['t'].append(t)
            self.stats[key]['x'].append(x)
            self.stats[key]['u'].append(u)
            self.stats[key]['c'].append(c)
            self.stats[key]['i'].append(i)
        return self.stats[key]

    
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
            times = np.array(stat['t']).flatten()            
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

    def plot_boxplots(self, path: Optional[str] = None,
                      filename: str = 'times.pdf',
                      title_prefix: Optional[str] = None,
                      plot_infeasibility: bool = False) -> None:
        """
        Makes a histogram of the times taken for each controller.
        """
        # plt.style.use('bmh')
        path = '.' if path is None else path
        
        n = len(self.stats)
        fig, axs = plt.subplots(1, 2, figsize=(3*n, 5))
        
        T = self.steps
        mpc_costs = self.stats[f'MPC_{T}']['c']
        for k, v in self.stats.items():
            # v['t_flatten'] = np.log(np.array(v['t']).flatten() * 1000) # convert to ms
            v['t_flatten'] = np.array(v['t']).flatten() * 1000  # convert to ms
            if k == f'MPC_{T}':
                v['gap'] = []                
            else:
                # v['gap'] = [(J - mpc_costs[i]) for i, J in enumerate(v['c']) ]
                v['gap'] = [(J - mpc_costs[i])/(mpc_costs[i] + 1e-6) for i, J in enumerate(v['c']) ]
            
        print(f"MPC gaps are:")
        print(self.stats[f'MPC_{T}']['gap'])
        what_to_plot = ['t_flatten', 'gap']
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
            ax.tick_params(axis='x', labelsize='large')
            ax.tick_params(axis='y', labelsize='large')
            if i == 0:          
                # ax.set_yscale('linear')
                # # Get the exponents
                # exp_t_min = np.floor(min([min(self.stats[key]['t_flatten']) for key in self.stats.keys()]))
                # exp_t_max = np.ceil(max([max(self.stats[key]['t_flatten']) for key in self.stats.keys()]))
                # # Set the y-ticks
                # yticks = np.arange(exp_t_min, exp_t_max + 1, 1)
                # ax.set_yticks(yticks)
                # ax.set_yticklabels([rf"$10^{{{int(exp)}}}$" for exp in yticks])

                ax.set_ylabel('Time (ms)', fontsize='large')
                ax.set_title("Per-step Computation Time", fontsize='large')
            else:
                # ax.set_yscale('log')
                ax.set_ylabel('Gap', fontsize='large')
                ax.set_title("Empirical normalized gap", fontsize='large')
                # ax.set_ylim(bottom=1e-6)
        M = len(self.stats[f'MPC_{T}']['c'])
        
        # title = f"Statistics over M={M} trajectories, horizon T={T}"
        # title = capitalize(title_prefix) + title if title_prefix is not None else title
        title = capitalize(title_prefix)
        plt.suptitle(title, fontsize='large')
        plt.tight_layout()
        plt.savefig(os.path.join(path, filename))
        
        # Plotting number of infeasible steps in the trajectory
        if plot_infeasibility:
            fig, ax = plt.subplots(1, 1, figsize=(1.5*n, 5))
            bp = ax.boxplot([self.stats[key]['i'] for key in self.stats.keys()],
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
            ax.set_ylabel('Infeasible events', fontsize='large')
            title = f"Infeasible events in a trajectory; M={M} trajectories, horizon T={T}"
            title = capitalize(title_prefix) + title if title_prefix is not None else title
            ax.set_title(title, fontsize='large')
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'infeas_' + filename))


    
    def plot_tradeoff(self, path: Optional[str] = None, filename: str = 'tradeoff.pdf', title: Optional[str] = None) -> None:
        """
        Plots the trade-off between online computation times (x-axis)
        and cost-to-go (y-axis) for each controller family (MPC and NN)"""
        path = '.' if path is None else path
        T = self.steps
        assert 'gap' in self.stats[f'MPC_{T}'], "Gaps not computed yet, run plot_boxplots first."
        plt.style.use('bmh')

        keys = list(self.stats.keys())
        nn_controllers = [k for k in keys if not k.startswith('MPC')]
        mpc_controllers = [k for k in keys if k.startswith('MPC')]

        nn_stats = []
        mpc_stats = []
        mint_stats = []

        self.stats[f'MPC_{T}']['gap'] = [0]
        for key in keys:
            t = np.median(self.stats[key]['t']) * 1000  # convert to ms
            gap = np.median(self.stats[key]['gap'])
            print(f"Controller {key}: median time {t:.4f}s, median gap {gap:.4f}")
            if key.startswith('MPC'):
                # Don't plot 'best' mpc
                # if key != f'MPC_{T}':
                mpc_stats.append((t, gap))
            elif key.startswith('NN'):
                nn_stats.append((t, gap))
            elif key.startswith('MINT'):
                mint_stats.append((t, gap))
            else:
                raise KeyError

        plt.figure(figsize=(7, 5))
        plt.plot(*zip(*mpc_stats), 's-', c='C0', label='MPC', markersize=13, linewidth=2)
        plt.plot(*zip(*nn_stats), '*-', c='C1', label='NPP (Ours)', markersize=16, linewidth=2)
        if len(mint_stats) > 0:
            plt.plot(*zip(*mint_stats), 'd-', c='olivedrab', label='MINT (Ours)', markersize=16, linewidth=2)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel("Per-step Computation Time (ms)", fontsize='large')
        plt.ylabel("Normalized Gap", fontsize='large')
        # title =  if title is None else title
        plt.title(title if title is not None else "Trade-off between computation time and cost-to-go", fontsize='large')
        plt.legend(fontsize='large')
        plt.tight_layout()
        plt.savefig(os.path.join(path, filename))
        # plt.show()
        
    
    def plot_trajectories(self, path: Optional[str] = None, filename: str = 'trajectories.pdf') -> None:
        """
        Plots all the trajectories for each controller.
        """
        plt.style.use('bmh')
        plt.figure()
        for i, (key, stat) in enumerate(self.stats.items()):
            for traj in stat['x']:
                plt.plot(traj[:, 0], traj[:, 1], 'o-', c=f"C{i}", markersize=5, alpha=0.5, label=key if traj is stat['x'][0] else None)  
        
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(f'Trajectories for different controllers')
        plt.legend()
        plt.tight_layout()
        if path is None:
            path = '.'
        plt.savefig(os.path.join(path, filename))
        # plt.show()

    def dump_stats(self, path: str, filename: str) -> None:
        """
        Dumps the statistics to a JSON file.
        """
        with open(os.path.join(path, filename), 'wb') as f:
            pickle.dump(self.stats, f)

if __name__ == "__main__":
    from constructor import constructor
    from data_collector import DataCollector
    from config import get_default_kwargs_yaml
    from utils import capitalize

    plt.rcParams.update({
    'axes.titlesize': 'large',
    'axes.labelsize': 'large',
    'xtick.labelsize': 'large',
    'ytick.labelsize': 'large',    
    })

    # Define the system and data collector
    env = 'pendulum'  # 'min_time'    
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    print(f"Configs are {cfgs}")

    # Path to save files -> Create based on time.
    hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results',
                        env,
                        hms_time)    
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'config.json'), encoding='utf-8', mode='w') as f:
                f.write(cfgs.tojson())
    
    # Create the system model and MPC controllers
    model, mpcs, simulator = constructor(env, cfgs)
    mpc_t = mpcs[0]  # Full horizon MPC
    collector = DataCollector(model, mpc_t, simulator, cfgs)
        
    # Grid anchors per dim.
    G = cfgs.G
    regressors = [NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0]) for _ in G]
    if cfgs.test_mint:
        regressors += [NNPolicy(nx=model.x.shape[0], nu=model.u.shape[0], k=10) for _ in G] 
        G += G

    print(f"Overall G is {G}")
    
    if hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub'):
        ub = np.array(cfgs.x_ub)
    else:
        ub = 2.0
    # method = 'grid'
    # for nn, g in zip(regressors, G):
    #     print(f"Regressor's name before adding data {nn.name}")
    #     # Collect data uniformly.
    #     data = collector.collect_data(num_trajectories=g**2, lb=-ub, ub=ub, method=method)
    #     if nn.name.startswith('NN') or nn.name.startswith('NPP'):
    #         nn.add_data(data['x'], data['u'])
    #     else:
    #         nn.add_data(data['x'], data['u'], data['J'])
    #     filename = f"{env}_{method}{g}_T{cfgs.N}_H{cfgs.mpc.n_horizon}_D{nn.size}.pkl"        
    #     collector.save_data(path=os.path.join(path, 'datasets'), filename=filename)
    # # This is optional!! Clear the data of previous iterations.
    #     collector.clear_data()
    #     print(f"Regressor's name after adding data {nn.name}")
    
    dir = "/Users/agu/Documents/Pycharm/npmpc/results/pendulum/2025-11-07-19-32-30/datasets"
    for nn, G in zip(regressors, G):
        with open(os.path.join(dir, f"pendulum_grid{G}_T100_H100_D2500.pkl"), 'rb') as fp:
            data = pickle.load(fp)
            if nn.name.startswith('NPP'):
                nn.add_data(data['x'], data['u'])

        

    


    M = cfgs.M  # Number of trajectories to evaluate 
    X0 = np.random.uniform(-ub, ub, size=(M, model.x.shape[0]))
    X0 = X0.reshape((M, model.x.shape[0], 1))    
    sampler = Sampler(X0=X0)

    evaluator = Evaluator(mpc_t)
    # Evaluate all the MPC controllers (full horizon first)
    for i, mpc in enumerate(mpcs):
        if i == 0 and hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub'):
            # Skipping the MPC teacher for the conservative problem
            continue  
        evaluator.evaluate(model, mpc, simulator, cfgs, sampler, M)        
    # # Evaluate receding horizon MPC controller
    # evaluator.evaluate(model, mpc_h, simulator, cfgs, sampler, M)        
    # Evaluate NN controller for different k
    for nn in regressors:
        # nn.set_k(k)
        evaluator.evaluate(model, nn, simulator, cfgs, sampler, M)
    
    # Plot results.
    try:
        size = nn.size
    except NameError:        
        size = 0
    
    # We plot infeasibility only if the config file has state bounds.
    plot_infeasibility = hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub')

    evaluator.plot_boxplots(path=path, filename=f'bp_{env}_d_{size}_M_{M}.pdf', title_prefix=f"{capitalize(env)}: ",
                            plot_infeasibility=plot_infeasibility)
    evaluator.plot_tradeoff(path=path, filename=f'tradeoff_{env}_d_{size}_M_{M}.pdf',
                            title=f"{capitalize(env)}: Computation Time/Cost-to-go trade-off")
    evaluator.plot_trajectories(path=path, filename=f'trajectories_{env}_d_{size}_M_{M}.pdf')

    evaluator.dump_stats(path=path, filename=f'stats_{env}_d_{size}_M_{M}.pkl')



