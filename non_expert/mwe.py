"""
Minimal Working Example of a policy iteration algorithm based on Bellman inequalities.

Date: 04/14/2026
Author: Agustin Castellano (@agucaste)

"""

import os
import time
import numpy as np

from core.config import Config, load_yaml
from core.nn_policy import MINTPolicy
from non_expert.helpers import J_upper_bound, Q_upper_bound, find_fixed_point_v2, get_cost_to_go
from non_expert.toy_example import ToySystem


if __name__ == "__main__":

    # Load environment configurations
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, "toy_config.yaml")
    config = Config.dict2config(load_yaml(cfg_path)["defaults"])

    # Get parameters
    x_ub = config.x_ub
    u_ub = config.u_ub
    a = config.a
    
    gamma = config.gamma
    lambd = config.lambd

    n = config.M0

    # Set up folder to save results
    hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    results_path = os.path.join(path, 'results')    
    os.makedirs(results_path, exist_ok=True)

    env = ToySystem(a=a, x_ub=x_ub)    
    
    X0 = np.random.uniform(-x_ub, x_ub, size=(n, 1))
    U0 = np.random.uniform(-x_ub, x_ub, size=(n, 1))
    D = [(x, u, env.c(x, u), env.f(x, u)) for x, u in zip(X0, U0)]

    Q = find_fixed_point_v2(D, gamma=gamma, lambd=lambd)
    print
    print(f"Q values:\n{Q}")

    # Plot to see what they look like.

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
    x = np.linspace(-x_ub, x_ub, 100)
    J_opt = env.cost_to_go(x, behavior='optimal', gamma=gamma)
    J_nothing = env.cost_to_go(x, behavior='nothing', gamma=gamma)

    # Prune indices
    J_ub = J_upper_bound(X0.ravel(), X0.ravel(), Q, lambd)
    to_prune = np.where(Q > J_ub)[0]

    c = np.zeros(n, dtype=object)
    c[:] = 'C0'  # Default color
    c[to_prune] = 'magenta'
    
    ax = axs[0]
    ax.scatter(X0, Q, label="Fixed point", marker="o", edgecolor=c)
    # ub = upper_bound(x, X0.ravel(), Q, lambd)
    # print(f'ub has shape {ub.shape}')
    # raise Exception
    ax.plot(x, J_upper_bound(x, X0.ravel(), Q, lambd), c='C0')
    ax.plot(x, J_opt, label=r"Optimal policy", color="black", linestyle="-.")
    ax.plot(x, J_nothing, label=r"Do nothing policy", color="forestgreen", linestyle="--")

    ax.set_xlabel("x")
    ax.set_ylabel("Q")
    ax.set_title("Fixed Point Q Values ; random data")
    ax.legend()

    


    # Evaluate policy
    pi_mint = MINTPolicy(nx=1, nu=1, k=100, lambd=lambd)
    pi_mint.add_data(X0, U0, Q)
    J_pi = []
    for x0 in x:
        J = get_cost_to_go(pi_mint, env, x0, gamma)
        J_pi.append(J)

    ax = axs[1]
    ax.set_xlabel("x")
    ax.set_ylabel("Cost-to-go")
    ax.set_title("Cost-to-go of MINT policy")

    ax.plot(x, J_upper_bound(x, X0.ravel(), Q, lambd), label='Upper bound')    
    ax.plot(x, J_pi, 's-', alpha=0.5, label='MINT; all data', color='indianred')
    
    
    # Evaluate MINT policy but dropping 'not useful' data.
    # pi_mint = MINTPolicy(nx=1, nu=1, k=100, lambd=lambd)
    X0_pruned = np.delete(X0, to_prune, axis=0)
    U0_pruned = np.delete(U0, to_prune, axis=0)
    Q_pruned = np.delete(Q, to_prune, axis=0)
    # pi_mint.add_data(X0_pruned, U0_pruned, Q_pruned)
    # J_pi = []
    # for x0 in x:
    #     J = simulate_cost_to_go(pi_mint, env, x0, gamma)
    #     J_pi.append(J)

    ax.scatter(X0_pruned, Q_pruned, marker="o", edgecolor='C0')

    # ax.plot(x, J_pi, 'd-', alpha=0.5, c='magenta', label='MINT; pruned data')
    
    ax.plot(x, J_opt, label=r"Optimal policy", color="black", linestyle="-.")

    ax.legend()

    


    X0 = X0_pruned 
    U0 = U0_pruned
    Q = Q_pruned




    ax = axs[2]
    ax.plot(x, J_opt, label=r"Optimal policy", color="black", linestyle="-.")
    ax.plot(x, J_upper_bound(x, X0.ravel(), Q, lambd), c='C0', linestyle='--', label='J_ub (0)')    
    ax.plot(x, J_pi, 'd-', alpha=0.5, c='C0', label='MINT (0)')


    # Now: Iterate by adding more data points
    M = config.M
    x_new = np.random.uniform(-x_ub, x_ub, size=(M, 1))
    u_new = np.zeros_like(x_new)
    c_new = np.zeros_like(x_new)
    x_next_new = np.zeros_like(x_new)
    
    for i, x0 in enumerate(x_new):
        u = pi_mint.make_step(x0.reshape(1, -1)) + np.random.normal(scale=config.sigma, size=(1, 1))  # Add some noise for exploration
        u = np.clip(u, -u_ub, u_ub)
        u_new[i] = u.squeeze()
        c_new[i] = env.c(x0, u)
        x_next_new[i] = env.f(x0, u)


    D_new = [(x, u, c, x_next) for x, u, c, x_next in zip(x_new, u_new, c_new, x_next_new)]

    # Only keep points where (TQ)(xl, ul) < Q(xl, ul)
    Q_new = J_upper_bound(x_new.ravel(), u_new.ravel(), X0.ravel(), U0.ravel(), Q, lambd)
    TQ_new = c_new.ravel() + gamma * J_upper_bound(x_next_new.ravel(), X0.ravel(), Q, lambd)  # This is the same as env.c + gamma * J_ub for the next state.
    
    improves = TQ_new < Q_new

    # print(f"TQ-Q for new points:\n{TQ_new - Q_new}")
    print('Negative entries of TQ_new - Q_new: ')
    print(np.sum(improves), '/', len(TQ_new))

    # Build a new dataset with the concatenation of the old data and the new data that satisfies the condition.
    X0 = np.concatenate([X0, x_new[improves]], axis=0)
    U0 = np.concatenate([U0, u_new[improves]], axis=0)    
    D = [(x, u, env.c(x, u), env.f(x, u)) for x, u in zip(X0, U0)]
    Q = find_fixed_point_v2(D, gamma=gamma, lambd=lambd)


    # print(f"New Q values:\n{Q}")
    print(f"New dataset size: {len(X0)}")

    # Define new policy and evaluate it.
    pi_mint = MINTPolicy(nx=1, nu=1, k=100, lambd=lambd)
    pi_mint.add_data(X0, U0, Q)
    J_pi = []
    for x0 in x:
        J = get_cost_to_go(pi_mint, env, x0, gamma)
        J_pi.append(J)
    
    ax.plot(x, J_upper_bound(x, X0.ravel(), Q, lambd), c='C1', linestyle='--', label='J_ub (1)')
    ax.plot(x, J_pi, 'd-', alpha=0.5, c='C1', label='MINT (1)')

    ax.set_title("Cost-to-go of MINT policy across iterations")
    ax.set_xlabel("x")
    ax.set_ylabel("Cost-to-go")
    ax.legend()



    plt.savefig(os.path.join(results_path, f"{hms_time}_a{a}_lambda{lambd}.pdf"))
