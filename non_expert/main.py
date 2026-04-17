import os, sys
import time
from matplotlib import pyplot as plt
import numpy as np

from core.config import Config, load_yaml
from core.nn_policy import MINTPolicy
from non_expert.helpers import J_upper_bound, Q_upper_bound, Tee, find_fixed_point_v2, get_cost_to_go
from non_expert.toy_example import ToySystem


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
k = config.k
q_tol = config.q_tol
q_iter = config.q_iter
max_iter = config.max_iter

M0 = config.M0  # Number of initial samples
M = config.M  # Number of samples for subsequent iterations
sigma = config.sigma  # Noise level for sampling new points

# Set up folder to save results
hms_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
results_path = os.path.join(path, 'results', hms_time)    
os.makedirs(results_path, exist_ok=True)

# Save config file
with open(os.path.join(results_path, 'config.json'), encoding='utf-8', mode='w') as f:
    f.write(config.tojson())

# Setup logging to file
output_file = open(os.path.join(results_path, "output.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, output_file)


# Create environment
env = ToySystem(a=a, x_ub=x_ub)    

# ------------------------
# Algorithm initialization
# ------------------------
#   1. Sample points uniformly at random
#   2. Find fixed point of T^Q
#   3. Prune unused data.

X0 = np.random.uniform(-x_ub, x_ub, size=(M0, 1))
U0 = np.random.uniform(-u_ub, u_ub, size=(M0, 1))
D = [(x, u, env.c(x, u), env.f(x, u)) for x, u in zip(X0, U0)]

Q = find_fixed_point_v2(D, gamma=gamma, lambd=lambd, tol=q_tol, max_iter=q_iter)

# Prune indices
J_ub = J_upper_bound(X0.ravel(), X0.ravel(), Q, lambd)
to_prune = np.where(Q > J_ub)[0]

X0, U0, Q = np.delete(X0, to_prune, axis=0), np.delete(U0, to_prune, axis=0), np.delete(Q, to_prune, axis=0)
print(f"After pruning, we have {len(X0)} samples left.")

# Define greedy policy
pi_mint = MINTPolicy(nx=1, nu=1, k=k, lambd=lambd)
pi_mint.add_data(X0, U0, Q)

# Evaluate on these states
x_eval = np.linspace(-x_ub, x_ub, 100)
J_opt = env.cost_to_go(x_eval, behavior='optimal', gamma=gamma)
J_pi = [get_cost_to_go(pi_mint, env, x, gamma) for x in x_eval]
J_ub = J_upper_bound(x_eval, X0.ravel(), Q, lambd)

len_D = np.zeros(max_iter)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
ax = axs[0]
ax.plot(x_eval, J_opt, label='Optimal Cost-to-Go', color='black', linestyle='--')
ax.plot(x_eval, J_pi, 's-', markersize=3, alpha=0.5, label='J_pi(0)', c='C0')
# ax.plot(x_eval, J_ub, c='C0')

# ax = axs[1]
axs[1].plot(x_eval, J_opt, label='Optimal Cost-to-Go', color='black', linestyle='--')
# axs[1].plot(x_eval, J_ub, c='C0')




for t in range(1, max_iter+1):
    x_new = np.random.uniform(-x_ub, x_ub, size=(M, 1))
    u_new = np.zeros_like(x_new)
    c_new = np.zeros_like(x_new)
    x_next_new = np.zeros_like(x_new)

    for i, x0 in enumerate(x_new):
        u = pi_mint.make_step(x0.reshape(1, -1)) + np.random.normal(scale=sigma, size=(1, 1))  # Add some noise for exploration
        u = np.clip(u, -u_ub, u_ub)
        u_new[i] = u.squeeze()
        c_new[i] = env.c(x0, u)
        x_next_new[i] = env.f(x0, u)

    Q_new = Q_upper_bound(x_new.ravel(), u_new.ravel(), X0.ravel(), U0.ravel(), q=Q, lambd=lambd)
    TQ_new = c_new.ravel() + gamma * J_upper_bound(x_next_new.ravel(), X0.ravel(), Q, lambd)  # This is the same as env.c + gamma * J_ub for the next state.
    improves = TQ_new < Q_new
    print(f"t={t}, number of improving points: {np.sum(improves)}/{len(TQ_new)}")

    X0 = np.concatenate([X0, x_new[improves]], axis=0)
    U0 = np.concatenate([U0, u_new[improves]], axis=0)
    D = [(x, u, env.c(x, u), env.f(x, u)) for x, u in zip(X0, U0)]
    Q = find_fixed_point_v2(D, gamma=gamma, lambd=lambd, tol=q_tol, max_iter=q_iter, warm_Q=Q)

    # Prune indices
    J_ub = J_upper_bound(X0.ravel(), X0.ravel(), Q, lambd)
    to_prune = np.where(Q > J_ub)[0]

    X0, U0, Q = np.delete(X0, to_prune, axis=0), np.delete(U0, to_prune, axis=0), np.delete(Q, to_prune, axis=0)
    print(f"t={t}, After pruning, we have {len(X0)} samples left.")

    len_D[t-1] = len(X0)


    if t % (config.max_iter // 5) == 0:
        # Define new policy and evaluate it.
        pi_mint = MINTPolicy(nx=1, nu=1, k=100, lambd=lambd)
        pi_mint.add_data(X0, U0, Q)
        J_pi = [get_cost_to_go(pi_mint, env, x, gamma) for x in x_eval]
        J_ub = J_upper_bound(x_eval, X0.ravel(), Q, lambd)

        ax.plot(x_eval, J_pi, 's-', markersize=3, alpha=0.5, label=f'J_pi({t})')  #, c=f'C{t}')
        axs[1].plot(x_eval, J_ub, linestyle='--', label=f'J_pi({t})')

    # plt.show()

for ax in [axs[0], axs[1]]:
    ax.set_xlabel("x")
    ax.legend()
axs[0].set_title("Cost-to-go of MINT policy across iterations")    
axs[0].set_ylabel("Cost-to-go")
axs[1].set_title("Upper bound across iterations")

axs[2].plot(len_D)
axs[2].set_title("Size of dataset (after pruning)")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("Number of samples")
axs[2].grid(alpha=0.5)

plt.savefig(os.path.join(results_path, f"{hms_time}_a{a}_lambda{lambd}.pdf"))






