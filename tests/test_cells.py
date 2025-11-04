import os
import sys

# Get absolute path to the folder *above* the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
CORE_DIR = os.path.join(PARENT_DIR, 'core')

# Add to sys.path if not already present
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import numpy as np
import time
from matplotlib import pyplot as plt

from tqdm import trange

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from nn_policy import NNRegressor
from config import Config
from data_collector import run_trajectory, get_trajectory_cost, count_infeasible_steps

from typing import Any, Callable, Optional
from numpy.typing import NDArray
import warnings

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
env = 'constrained_lqr_2'  # 'min_time'    
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
    
# G = [5, 9, 11, 15]  # [5, 7, 9, 11]  # [3, 5, 7, 9, 11]  # grid anchors per dimension
G = cfgs.G
G = []
regressors = [NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0]) for _ in G]

if hasattr(cfgs, 'x_lb') and hasattr(cfgs, 'x_ub'):
    ub = np.array(cfgs.x_ub)
else:
    ub = 2.0
method = 'grid'
for nn, g in zip(regressors, G):
    # Collect data uniformly.
    data = collector.collect_data(num_trajectories=g**2, lb=-ub, ub=ub, method=method)
    nn.add_data(data['x'], data['u'])

    filename = f"{env}_{method}{g}_T{cfgs.N}_H{cfgs.mpc.n_horizon}_D{nn.size}.pkl"        
    collector.save_data(path=os.path.join(path, 'datasets'), filename=filename)

    




M = 100  # Number of trajectories to evaluate 
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



