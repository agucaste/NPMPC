"""
Toy example for building Bellman inequalities. Given a behavior policy π, we collect a dataset of (xi, ui, ci, xi') transitions, and then fit a nonparametric Q function
that satisfies the Bellman inequality:
    ci + γ * (T^π J)(xi') <= J(xi) - ε,  for all i
where  J(x) = min_i { Q(xi) + λ * dist(x, xi) } is a nonparametric 'upper' envelope of the value function.

System:
    x' = x + u
    c(x, u) = |x|

Policies:
    - π(x) = 0 (do nothing)
    - π(x) = -x (optimal)

Author: Agustin Castellano (@agucaste)
03/25/26
"""

import os

from matplotlib import pyplot as plt
import numpy as np

from non_expert.helpers import J_upper_bound, solve_optimization_problem


class ToySystem:
    """
    Toy system, with dynamics
    x' = a*x + u
    c = |x|
    """
    def __init__(self, a, x_ub):
        self.a = a
        self.x_ub = x_ub

        self.Lf = np.abs(a)
        self.Lc = 1.0

    def f(self, x, u):
        """
        Transition function
        """
        return self.a * x + u
    
    def c(self, x, u):
        """
        Stage cost
        """
        return np.abs(x)
    
    def cost_to_go(self, x: np.ndarray, behavior: str, gamma: float):
        """
        Computes the cost-to-go J^pi(x) for a given policy pi using value iteration.
        """
        
        J = np.array([self.c(xi, 0) for xi in x])  # Optimal cost-to-go.
        if behavior == 'nothing':
            if self.a <= 1:
                J = J / (1 - self.a * gamma)  # True J values for "do nothing" policy
            else:
                # System will expand, then hit the boundary.
                hitting_time = np.ceil(np.log(self.x_ub / np.abs(x)) / np.log(self.a))
                J = np.where(hitting_time >= 0, J * (1 - (gamma*self.a)**hitting_time) / (1 - self.a * gamma) + self.x_ub * gamma**hitting_time / (1 - gamma), J/(1 - self.a * gamma))        
        return J
    
    # def reset(self, x0: Optional[np.ndarray] = None):
    #     """
    #     Resets the system to a given state.
    #     """
    #     if x0 is not None:
    #         assert x0.shape == (1,), "x0 should be a 1D array with shape (1,)"
    #     self.x = x0
    #     return self.x
    

"""
Main code begins here
"""
if __name__ == "__main__":
    # -----------------
    # System parameters
    # -----------------
    gamma = 0.9  
    x_ub = 2
    n = 15  # Number of samples.
    a = 1.1  # System growth.
    lambda_min = 50

    # -----------------------
    # Optimization parameters
    # -----------------------
    w_eps, w_lambda = 1.0, 1.0  # Weights for epsilon and lambda in the objective
    epsilons = [0, 0.5, 1]    

    # --------
    # Sampling
    # --------
    sampling = 'random'  # 'random' or 'uniform'

    # -----------
    # Environment
    # -----------
    env = ToySystem(a=a, x_ub=2)

    # ---------------
    # Behavior Policy
    # ---------------
    behavior = 'nothing'  # 'nothing' or 'optimal'
    pi = (lambda x: 0) if behavior == 'nothing' else (lambda x: -env.a * x)



    for trial in range(10):
        X0 = np.sort(np.random.uniform(-x_ub, x_ub, n))  if sampling == 'random' else np.linspace(-x_ub, x_ub, n)

        # Collect transitions
        D = []
        for x in X0:
            u = pi(x)
            D.append((x, u, env.c(x, u), env.f(x, u)))  # Collect (x, u, c, x')

        # Get costs
        x = np.linspace(-x_ub, x_ub, 100)
        J_b = env.cost_to_go(x, behavior=behavior, gamma=gamma)
        print(f"Baseline J_b: {J_b}")
        


        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        ax = axs[0]
        ax.plot(x, J_b, label=r"True $J^{\pi_\beta}$", color="black")

        
        for e_min in epsilons:
            result = solve_optimization_problem(D, e_min=e_min, lambda_min=lambda_min, verbose=True)
            ax.scatter(X0, result["q"], label=rf"$J^{{ub}},~\varepsilon_{{min}}={e_min}$", marker="o")
            # ax.plot(x, [upper_bound(xi, X0, result["q"], result["lambda"]) for xi in x])  #  label=rf"Upper Bound, $\varepsilon_{{min}}={e_min}$", linestyle="--")
            ax.plot(x, J_upper_bound(x, X0, result["q"], result["lambda"]))  #  label=rf"Upper Bound, $\varepsilon_{{min}}={e_min}$", linestyle="--")
            # # print(result["dist"][:, 2])
            # print(f"Q values:\n{result['q']}")
            # print(f"Epsilon: {result['epsilon']}")
            # print(f"Lambda: {result['lambda']}")
            
            bellman_residual = -result["q"] + np.array([env.c(xi, 0) for xi in X0]) + gamma * result["J_next"]
            axs[1].plot(X0, bellman_residual, label=rf"$(T^{{\pi}}J)(x_i) - J(x_i)$, $\varepsilon_{{min}}={e_min}$", marker="o", linestyle="--")

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"Value")
        axs[1].set_xlabel(r"$x$")
        axs[1].set_ylabel(r"Bellman Residual")
        title = "X0 = [" + ", ".join([f"{x:.2f}" for x in X0]) + "]"
        ax.set_title(title)
        ax.legend()
        axs[1].legend()
        plt.suptitle(f"a: {env.a}, π: {behavior}, λ: {result["lambda"]}, n: {n}")
        plt.tight_layout()

        save_folder = f"./non-expert/a{env.a}_{behavior}_l{result['lambda']}_n{n}"
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(f"{save_folder}/toy_example_x0_{trial}.pdf")
        # plt.show()

        if sampling == 'uniform':
            break  # Only need to run one trial
