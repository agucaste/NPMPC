"""

Author: Agustin Castellano (@agucaste)
This script loads a series of datasets and tests the NNRegressor on each of them.

"""
import os, pickle, time
import numpy as np
from config import Config, get_default_kwargs_yaml, load_yaml
from constructor import constructor
from evaluator import Sampler, Evaluator
from nn_policy import MINTPolicy, NNRegressor
from utils import capitalize

# Pendulum results for paper
path = "/Users/agu/Documents/Pycharm/npmpc/results/pendulum/2025-11-07-19-32-30/datasets"
files = [os.path.join(path, d) for d in
         ["pendulum_grid5_T100_H100_D2500.pkl",
          "pendulum_grid7_T100_H100_D4900.pkl",
          "pendulum_grid9_T100_H100_D8100.pkl",
          "pendulum_grid11_T100_H100_D12100.pkl"]]

# Minimum time results for paper
path = "/Users/agu/Documents/Pycharm/npmpc/results/min_time/2025-11-06-21-15-21/datasets"
files = [os.path.join(path, d) for d in
         [
             "min_time_grid3_T200_H200_D1800.pkl",
             "min_time_grid5_T200_H200_D5000.pkl",
             "min_time_grid7_T200_H200_D9800.pkl",
             "min_time_grid9_T200_H200_D16200.pkl"
         ]]


# path = "/Users/agu/Documents/Pycharm/npmpc/results/smart_sampling/pendulum/2025-11-09-14-07-50/datasets/"
# path = "/Users/agu/Documents/Pycharm/npmpc/results/smart_sampling/pendulum/2025-11-10-10-04-48/datasets"

path = "/Users/agu/Documents/Pycharm/npmpc/results/smart_sampling/pendulum/2025-11-20-13-31-39/datasets"

path = "/Users/agu/Documents/Pycharm/npmpc/results/smart_sampling/min_time/2025-11-20-13-31-57/datasets"

env = 'min_time'
# cfgs = get_default_kwargs_yaml(algo='', env_id=env)
cfgs = Config().dict2config(load_yaml(os.path.join(path, '..', 'config.json')))
cfgs.mpc.n_horizons = [100, 50, 20, 10]  # adding '10' to see if plots look better.

k = 10
lambd = 1e7


model, mpcs, simulator = constructor(env, cfgs)
mpc_t = mpcs[0]  # Full horizon MPC

regressors = []
i = 1
f = os.path.join(path, f'dataset_{i}.pkl')
while os.path.exists(f):
# for f in files:
    # with open(os.path.join(path, f"dataset_{i}.pkl"), 'rb') as fp:
    with open(f, 'rb') as fp:
        data = pickle.load(fp)
    # print(f"data is {data}")
    # Add the NN Regressor:
    nn = NNRegressor(nx=model.x.shape[0], nu=model.u.shape[0])
    nn.add_data(data['x'], data['u'])
    regressors.append(nn)    

    nn = MINTPolicy(nx=model.x.shape[0], nu=model.u.shape[0], k=k, lambd=lambd)
    nn.add_data(data['x'], data['u'], data['J'])
    regressors.append(nn)
    
    i += 1
    f = os.path.join(path, f'dataset_{i}.pkl')


hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
os.makedirs(os.path.join(path, '..', 'evaluations', hms_time), exist_ok=True)
path = os.path.join(path, '..', 'evaluations', hms_time)

print(f"how many regressors? {len(regressors)}")
M = 100  # cfgs.M  # Number of trajectories to evaluate 

ub = np.array(cfgs.ub_eval)
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

evaluator.plot_boxplots(path=path,
                        filename=f'bp_{env}_d_{size}_M_{M}.pdf',
                        title_prefix=f"{capitalize(env)}: ",
                        plot_infeasibility=plot_infeasibility)

# Plot tradeoffs with and without log scale.
evaluator.plot_tradeoff(path=path,
                        filename=f'tradeoff_{env}_d_{size}_M_{M}.pdf',
                        title=f"{capitalize(env)}: Computation Time/Cost-to-go trade-off"
                        )
evaluator.plot_tradeoff(path=path,
                        filename=f'tradeoff_log_{env}_d_{size}_M_{M}.pdf',
                        title=f"{capitalize(env)}: Computation Time/Cost-to-go trade-off",
                        log_y=True
                        )
evaluator.plot_trajectories(path=path, filename=f'trajectories_{env}_d_{size}_M_{M}.pdf')

evaluator.dump_stats(path=path, filename=f'stats_{env}_d_{size}_M_{M}.pkl')
