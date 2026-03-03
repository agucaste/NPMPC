"""
Author: Agustin Castellano (@agucaste)

A modified version of 'evaluator.py'.
    - Instead of sampling from a uniform grid, uses the "verification" algorithm (Algorithm 2) to get samples.
    - Parallelizes data collection
"""
from multiprocessing import Pool
import numpy as np
import time
from matplotlib import pyplot as plt

from tqdm import tqdm, trange
import os, sys
import pickle

from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator

from cell import Cell
from nn_policy import NNRegressor, NNPolicy
from config import Config
from data_collector import cell_verifier_worker_batch, collect_single_trajectory_serial, run_trajectory, get_trajectory_cost, count_infeasible_steps

from typing import Any, Callable, Optional
from numpy.typing import NDArray
import warnings

from constructor import constructor
from data_collector import DataCollector
from config import get_default_kwargs_yaml

from utils import capitalize




def save_data(path: str, filename: str, data: dict) -> None:
        """
        Saves the dataset to a file.
        Args:
            - path: directory path to save the file.
            - filename: name of the file.
        """
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {full_path}")
        return


if __name__ == '__main__':
    # Get absolute path to the folder *above* the current file's directory
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(CURRENT_DIR)
    TEST_DIR = os.path.join(PARENT_DIR, 'tests')
    # Add to sys.path if not already present
    if TEST_DIR not in sys.path:
        sys.path.insert(0, TEST_DIR)
    # print(f" Core dir is {TEST_DIR}")
    from test_cells import plot_cells, update_list_of_cells # type: ignore

    plt.rcParams.update({k: 'large' for k in ['axes.titlesize', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize']
    })

    # Create environment and directory
    # --------------------------------
    env = 'pendulum'  # 'min_time', 'pendulum' or 'constrained_lqr'    
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    model, mpcs, simulator = constructor(env, cfgs)

    # Path to save files -> Create based on time.
    hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'results',
                        'smart_sampling',
                        env,
                        hms_time)    
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'config.json'), encoding='utf-8', mode='w') as f:
        f.write(cfgs.tojson())
   
    # NUM_CORES = os.cpu_count()
    # with Pool(processes=NUM_CORES) as pool:
    #     chunks = np.array_split(X0, NUM_CORES)
    #     all_results = pool.starmap(worker_batch,
    #                                [(env, cfgs, chunk, cfgs.N, lb, ub) for chunk in chunks])
    # results = sum(all_results, [])  # flatten    
    # total_time = time() - total_time
    # print(f"Praying this works -> time: {total_time}")

    # data = {k: [r[k] for r in results] for k in results[0].keys()}



    # Begin here.
    mpc_t = mpcs[0]  # Full horizon MPC
    regressor = NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0])

    # This is where we will save the data.
    data = {k: [] for k in ['x', 'u', 't', 'c', 'J', 'i', 'd']}

    try:
        ub = cfgs.x_ub
    except AttributeError:        
        ub = cfgs.ub_eval
    ub = np.array(ub)

    # Get verification parameters.
    beta = cfgs.beta
    lambd = cfgs.lambd
    eta = cfgs.eta

    red_cells = [Cell(center=np.zeros_like(ub), radius=ub[0], split_factor=3)]
    green_cells = []

    if not cfgs.parallelize_cell_verification:
        # Only care about verifying 'optimality gap' (problem is unconstrained so feasibility is good)
        for t in range(10):
            print(f"Number of uncertified cells: {len(red_cells)}")
            if len(red_cells) == 0: 
                print("Finished! Breaking")
                break    
            # Can this be parallelized?
            for cell in tqdm(red_cells, desc='Verifying cells...'):            
                if not cell.is_certified:
                    traj_data = collect_single_trajectory_serial(cell.c.reshape(-1, 1), cfgs.N, -ub, ub, cfgs, model, mpc_t, simulator)                
                    for k in data.keys():
                        data[k].append(traj_data[k])
                    # Certify (or not) the cell:
                    J = traj_data['c']
                    # print(f"J is {J}")
                    r = beta/(2 + beta)/lambd*(J+eta)
                    # print(f"Cell radius: {cell.r:.2f}\t Certified: {r:.2f}")
                    cell.certify(r)
            plot_cells(red_cells+green_cells, path, f'optimality_{t}.pdf', f"Optimality; i={t}", cfgs, mode='optimality') 

            red_cells, green_cells = update_list_of_cells(red_cells, green_cells)
            save_data(path=os.path.join(path, 'datasets'), filename=f'dataset_{t}.pkl', data=data)

    else:
        # Use parallelization 
        for t in range(10):            
            print(f"Number of uncertified cells: {len(red_cells)}")
            if len(red_cells) == 0: 
                print("Finished! Breaking")
                break    
            with Pool(processes=6) as pool:
                cell_chunks = np.array_split(red_cells, 8)
                results = pool.starmap(
                    cell_verifier_worker_batch,
                    [(env, cfgs, chunk, cfgs.N, -ub, ub) for chunk in cell_chunks]
                )
            # Merge each worker’s results
            for worker_data in results:
                for k in data.keys():
                    data[k].extend(worker_data[k])
            # print(f"results are \n{results}")
            # for r in results:
                # for k in data.keys():
                    # data[k].append(r[k])
            # for k in data.keys():
            #     data[k].append(traj_data[k])
            plot_cells(red_cells+green_cells, path, f'optimality_{t}.pdf', f"Optimality; i={t}", cfgs, mode='optimality') 

            red_cells, green_cells = update_list_of_cells(red_cells, green_cells)
            save_data(path=os.path.join(path, 'datasets'), filename=f'dataset_{t}.pkl', data=data)
