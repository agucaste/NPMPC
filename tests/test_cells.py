from copy import deepcopy
import os
import sys

from matplotlib.patches import Rectangle

from core.cell import Cell

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

from typing import Any, Callable, List, Optional
from numpy.typing import NDArray
import warnings

import sys, os
from constructor import constructor
from data_collector import DataCollector
from config import get_default_kwargs_yaml
from utils import capitalize


def plot_cells(cell_collection: List[Cell], path: str, filename: str, title: str, cfgs: Config,
               mode: str = 'feasibility'):
    """
    Plots a list of cells, with colors corresponding to certification.
    """
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    ax.set_xlim(cfgs.x_lb[0], cfgs.x_ub[0])
    ax.set_ylim(cfgs.x_lb[1], cfgs.x_ub[1])
    ax.set_aspect('equal')
    # print("Plotting...")

    for cell in cell_collection:
        # Pick color
        if mode == 'feasibility':
             facecolor = (0, 1, 0, .3) if cell.is_certified else (1, 0, 0, .3)  # Green or Red
        elif mode == 'optimality':
            #  facecolor = (0, 0, 1, .3) if cell.is_certified else (0.8, 0.45, 0.3, .3)  # Blue or Gray
            facecolor = (0, 0, 1, .3) if cell.is_certified else (1, 1, .3, .3)  # Blue or Gray
        # Get center and radius        
        c, r = cell.c, cell.r
        # print(f"cell center: {c}\nradius: {r}\n\n")
        # South-west corner
        sw = c - r
        square = Rectangle(sw, 2*r, 2*r,
                        #    edgecolor=None,
                           edgecolor=(0, 0, 0, 1),  # Black: RGB 000, alpha=1
                           linewidth=.1,
                           facecolor=facecolor,
        )
        ax.add_patch(square)
    
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ticks = [cfgs.x_lb[0], 0, cfgs.x_ub[0]]
    ax.set_xticks(ticks, ticks)
    ax.set_yticks(ticks, ticks)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, filename))






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
                    'verification'
                    env,
                    hms_time)    
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, 'config.json'), encoding='utf-8', mode='w') as f:
            f.write(cfgs.tojson())

# Create the system model and MPC controllers
model, mpcs, simulator = constructor(env, cfgs)
mpc_t = mpcs[0]  # Full horizon MPC
collector = DataCollector(model, mpc_t, simulator, cfgs)

# Sanity check
assert max(cfgs.x_ub) == min(cfgs.x_ub)

regressor = NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0])

red_cells = [
    Cell(center=np.zeros_like(cfgs.x_ub), radius=cfgs.x_ub[0], split_factor=3)
]
green_cells = []

L_f = cfgs.L_f  # Lipschitz constant.
# while len(cell_colection) > 0:


for t in range(10):
    print(f"Number of uncertified cells: {len(red_cells)}")
    if len(red_cells) == 0: 
         print("Finished! Breaking")
         break    
    # 1st step: run a trajectory from each cell center
    for cell in red_cells:
        if not cell.is_certified:
            data = collector.collect_data(num_trajectories=1,
                                        lb=cfgs.x_lb,
                                        ub=cfgs.x_ub,
                                        method='start_from',
                                        x0=cell.c,
                                        disable_tqdm=True
                                        )
            # Distance to the unsafe region (aka radius)
            r = data['d'][0][0]
            # Certify cell
            cell.certify(r/L_f)
            collector.clear_data()    
    plot_cells(red_cells+green_cells, path, f'feasibility_{t}.pdf', f"Feasibility; i={t}", cfgs) 
    
    new_red_cells = []
    for cell in red_cells:
         if cell.is_certified:
              green_cells.append(cell)
         else:
              new_red_cells += cell.split()
    red_cells = new_red_cells


print(f"Testing suboptimality!")
assert len(red_cells)  == 0

red_cells = green_cells
for cell in red_cells:
    cell.certify(0)
    assert not cell.is_certified
# Empty green cells
green_cells = []


beta = cfgs.beta
lambd = cfgs.lambd
eta = cfgs.eta

for t in range(10):
    print(f"Number of uncertified cells: {len(red_cells)}")
    if len(red_cells) == 0: 
         print("Finished! Breaking")
         break    
    # 1st step: run a trajectory from each cell center
    for cell in red_cells:
        if not cell.is_certified:
            data = collector.collect_data(num_trajectories=1,
                                        lb=cfgs.x_lb,
                                        ub=cfgs.x_ub,
                                        method='start_from',
                                        x0=cell.c,
                                        disable_tqdm=True
                                        )
            # Distance to the unsafe region (aka radius)
            r = data['d'][0][0]
            J = data['c'][0]
            # Certify cell
            cell.certify(beta/(2 + beta)/lambd*(J+eta))
            collector.clear_data()    
    plot_cells(red_cells+green_cells, path, f'optimality_{t}.pdf', f"Optimality; i={t}", cfgs, mode='optimality') 
    
    new_red_cells = []
    for cell in red_cells:
         if cell.is_certified:
              green_cells.append(cell)
         else:
              new_red_cells += cell.split()
    red_cells = new_red_cells


    


    
    # all_cells = deepcopy(cell_collection)

    



    



      
    


