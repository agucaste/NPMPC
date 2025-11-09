"""
A script to test (potential) parallelization of the data collection.
"""


from copy import deepcopy
import os, sys

# Get absolute path to the folder *above* the current file's directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
CORE_DIR = os.path.join(PARENT_DIR, 'core')

# Add to sys.path if not already present
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from config import get_default_kwargs_yaml
from data_collector import collect_single_trajectory_serial, run_trajectory, collect_single_trajectory, worker_batch
import numpy as np
from constructor import constructor

if __name__ == '__main__':
    import warnings
    from multiprocessing import Pool
    from time import time

    
    NUM_CORES = os.cpu_count()
    print(f"Using {NUM_CORES} cores")

    env = 'constrained_lqr_2'
    cfgs = get_default_kwargs_yaml(algo='', env_id=env)
    lb, ub = cfgs.x_lb, cfgs.x_ub
    
    n = len(lb)
    M = 64  # 2048 * 4
    X0 = np.random.uniform(lb, ub, size=(M, n)).reshape(M, n, 1)
    # print(X)


    # args = [(env, X0[i], cfgs.N, lb, ub, cfgs) for i in range(M)]
    # # x0 must be shape (n, 1)
    # total_time = time()
    # with Pool(processes=NUM_CORES) as pool:
    #     results = pool.starmap(collect_single_trajectory, args)
    # # data = collect_single_trajectory(env, x0, cfgs.N, lb, ub, cfgs)
    # total_time = time() - total_time
    # print(f"Option 1: Parallel execution; system constructed in parallel\n\
    #       Total (parallel) execution time: {total_time:.2f}")


    total_time = time()
    model, mpcs, simulator = constructor(env, cfgs)
    mpc = mpcs[0]
    for i in range(M):
        collect_single_trajectory_serial(
            X0[i],
            cfgs.N,
            lb,
            ub,
            cfgs,
            model,
            mpc,
            simulator
        )
    total_time = time() - total_time
    print(f"Total (serial) execution time: {total_time:.2f}")


    # Go dirty!!! can we pass one instance of mpc, model, simulator to the parallelizer??

    # total_time = time()
    # # args = [(env, X0[i], cfgs.N, lb, ub, cfgs, deepcopy(model), deepcopy(mpc), deepcopy(simulator)) for i in range(M)]
    # # x0 must be shape (n, 1)
    # total_time = time()

    # Doesn't work!! -> Go by chunks.


    total_time = time()
    # args = [(env, X0[i], cfgs.N, lb, ub, cfgs, deepcopy(model), deepcopy(mpc), deepcopy(simulator)) for i in range(M)]
    # x0 must be shape (n, 1)
    num_cores = 8
    with Pool(processes=num_cores) as pool:
        chunks = np.array_split(X0, num_cores)
        all_results = pool.starmap(worker_batch,
                                   [(env, cfgs, chunk, cfgs.N, lb, ub) for chunk in chunks])
    results = sum(all_results, [])  # flatten    
    total_time = time() - total_time
    print(f"Praying this works -> time: {total_time}")

    data = {k: [r[k] for r in results] for k in results[0].keys()}
    for k in data.keys():
        print(f'key: {k}\nlength:{len(data[k])}')
    print(data[k])


