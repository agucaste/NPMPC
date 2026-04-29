from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from core.nn_policy import MINTPolicy, NNRegressor


class Tee:
    """Duplicate stdout writes to multiple file-like streams."""

    def __init__(self, *streams):
        """Store streams such as sys.stdout and an open log file."""
        self.streams = streams

    def write(self, data):
        """Write data to all streams."""
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            stream.flush()


def seed_all(seed: int) -> None:
    """Seed the RNGs used by the non-expert experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_cost_to_go(policy: NNRegressor | MINTPolicy, env, x0: np.ndarray, gamma: float):
    x = x0.reshape(1, -1)
    T = int(np.ceil(1 / (1 - gamma)))
    J = 0
    for t in range(T):
        u = policy.make_step(x)
        c = env.c(x, u)
        x = env.f(x, u)
        J += c * (gamma ** t)
    return J.item()


def J_upper_bound(distances: np.ndarray, q: np.ndarray, lambd):
    """Computes J_ub from query-to-dataset distances: min_k { Q_k + lambda * distance_k }."""
    q = np.asarray(q, dtype=float).reshape(-1)
    assert q.ndim == 1, "q should be a 1D array"
    if hasattr(lambd, "X"):
        lambd = lambd.X
    distances = np.asarray(distances, dtype=float)
    if distances.ndim == 1:
        distances = distances.reshape(1, -1)
    assert distances.shape[1] == q.shape[0], "distances columns must match q"
    return np.min(q[None, :] + lambd * distances, axis=1)


def Q_upper_bound(
    y: np.ndarray,
    u: np.ndarray,
    x_d: np.ndarray,
    u_d: np.ndarray,
    q: np.ndarray,
    lambd,
    batch_size: Optional[int] = None,
):
    """Computes Q_ub(y, u) from metric-space points y/x_d and action-space points u/u_d."""
    assert q.ndim == 1, "q should be a 1D array"
    if hasattr(lambd, "X"):
        lambd = lambd.X
    y = as_rows(y)
    u = as_rows(u)
    x_d = as_rows(x_d)
    u_d = as_rows(u_d)
    batch_size = _distance_batch_size() if batch_size is None else max(1, int(batch_size))
    out = np.empty(y.shape[0], dtype=float)

    for start in range(0, y.shape[0], batch_size):
        stop = min(start + batch_size, y.shape[0])
        dists_x = np.linalg.norm(x_d[:, None, :] - y[None, start:stop, :], axis=-1)
        dists_u = np.linalg.norm(u_d[:, None, :] - u[None, start:stop, :], axis=-1)
        out[start:stop] = np.min(q[:, None] + lambd * dists_x + lambd * dists_u, axis=0)

    return out


def find_fixed_point(D: List[Tuple], lambd: float, gamma: float, tol: float = 1e-3, max_iter: int = 1000):
    """
    Finds the fixed point of the Bellman operator T^Qub:

    (TQ)i = ci + gamma * min_k { Q(xk) + lambda * dist(xk, xi') }
    """
    raise DeprecationWarning("This function is deprecated. Use find_fixed_point_v2 instead.")
    assert len(D) > 0 and lambd > 0 and 0 < gamma < 1, "Invalid inputs"
    n = len(D)
    x, _, c, x_next = map(np.array, zip(*D))

    dist = np.abs(x.reshape(-1, 1) - x_next.reshape(1, -1))

    Q = np.zeros(n)
    for it in range(max_iter):
        Q_next = np.zeros_like(Q)
        for i in range(n):
            Q_next[i] = c[i] + gamma * np.min(Q + lambd * dist[:, i])
        if np.max(np.abs(Q_next - Q)) < tol:
            print(f"Converged after {it} iterations.")
            Q = Q_next
            break
        Q = Q_next
    return Q


def as_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def _distance_batch_size() -> int:
    """Return the query chunk size used for dense distance reductions."""
    return max(1, int(os.environ.get("MINT_DISTANCE_BATCH_SIZE", "1024")))


def _min_lipschitz_values(
    x_d: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    lambd: float,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """Computes min_i q_i + lambda * ||x_i - y_j|| in the provided metric space."""
    x_d = as_rows(x_d)
    y = as_rows(y)
    q = np.asarray(q, dtype=float).reshape(-1)
    assert x_d.shape[0] == q.shape[0], "q must match number of source samples"

    batch_size = _distance_batch_size() if batch_size is None else max(1, int(batch_size))
    out = np.empty(y.shape[0], dtype=float)

    for start in range(0, y.shape[0], batch_size):
        stop = min(start + batch_size, y.shape[0])
        dists = np.linalg.norm(x_d[:, None, :] - y[None, start:stop, :], axis=-1)
        out[start:stop] = np.min(q[:, None] + lambd * dists, axis=0)

    return out


def find_fixed_point_v2(
    D: List[Tuple],
    lambd: float,
    gamma: float,
    tol: float = 1e-3,
    max_iter: int = 1000,
    warm_Q: Optional[np.ndarray] = None,
    batch_size: Optional[int] = None,
):
    """
    Vectorized version of function `find_fixed_point`.
    Finds the fixed point of the Bellman operator T^Qub:

    (TQ)i = ci + gamma * min_k { Q(xk) + lambda * dist(xk, xi') }
    The x entries in D are points in the chosen metric space, e.g. encoded observations.
    """
    assert len(D) > 0 and lambd > 0 and 0 < gamma < 1, "Invalid inputs"

    x, _, c, x_next = zip(*D)

    x = as_rows(x)
    x_next = as_rows(x_next)
    c = np.asarray(c, dtype=float).reshape(-1)

    assert x.shape == x_next.shape, "x and x_next must have matching shapes"
    assert x.shape[0] == c.shape[0], "costs must match number of samples"

    Q = np.zeros(x.shape[0]) if warm_Q is None else np.concatenate([warm_Q, np.zeros(len(D) - len(warm_Q))])
    for it in range(max_iter):
        Q_next = c + gamma * _min_lipschitz_values(x, x_next, Q, lambd, batch_size=batch_size)

        if np.max(np.abs(Q_next - Q)) < tol:
            print(f"Value Iteration Converged after {it} steps.")
            return Q_next

        Q = Q_next

    return Q

def find_fixed_point_v3(
    D: List[Tuple],
    lambd: float,
    gamma: float,
    tol: float = 1e-3,
    max_iter: int = 1000,
    warm_Q: Optional[np.ndarray] = None,
    K: int = 5,
    batch_size: Optional[int] = None,
):
    r"""
    K-step bootstrapped version of `find_fixed_point_v2`.
    Solves value iteration with (optimistic) k-step bootstrapping, 1 <= k <= K.

    (TQ)_i = \min_{1 <= k <= K} c + γ*c' + γ^2*c'' + ... + γ^{k-1}*c'^{k-1} + γ^k * min_j { Q(xj) + λ*dist(xj, x'^{k}) }

    The x entries in D are points in the chosen metric space, e.g. encoded observations.

    Args:
        D (List[Tuple]): Dataset of transitions (xi, ui, ci, xi', ui', ci', xi'', ui'', ci'', ...).
        lambd (float): constant for the Lipschitz term in the Bellman operator.
        gamma (float): discount factor.
        tol (float, optional): tolerance for convergence (sup norm).
        max_iter (int, optional): maximum number of iterations. 
        warm_Q (Optional[np.ndarray], optional): initial guess for the Q-function.
        K (int, optional): number of steps for bootstrapping. Defaults to 5.

    Returns:
        Tuple[np.ndarray, float, float, int]: Q values, average selected bootstrap
        step, median selected bootstrap step across all value-iteration updates,
        and the number of value-iteration updates run.
    """
    assert len(D) > 0 and lambd > 0 and 0 < gamma < 1, "Invalid inputs"
    assert K >= 1, "K must be at least 1"
    assert all(len(d) == 3 * K + 1 for d in D), "Each transition tuple must contain (xi, ui, ci, xi', ui', ci', ..., xi^{K})"

    x = as_rows([d[0] for d in D])
    x_nexts = [as_rows([d[3 + 3*k] for d in D]) for k in range(K)]
    costs = [np.asarray([d[2 + 3*k] for d in D], dtype=float).reshape(-1) for k in range(K)]

    assert all(x_next.shape == x.shape for x_next in x_nexts), "all x_next arrays must match x shape"
    assert all(cost.shape[0] == x.shape[0] for cost in costs), "costs must match number of samples"

    discounted_costs = np.zeros((K, x.shape[0]))
    running_cost = np.zeros(x.shape[0])
    for k in range(K):
        running_cost = running_cost + (gamma ** k) * costs[k]
        discounted_costs[k] = running_cost

    if warm_Q is None:
        Q = np.zeros(x.shape[0])
    else:
        Q = np.concatenate([warm_Q, np.zeros(len(D) - len(warm_Q))])

    gamma_powers = gamma ** np.arange(1, K + 1)
    all_bootstrap_steps = []
    for it in range(max_iter):
        bootstraps = np.vstack(
            [
                _min_lipschitz_values(x, x_nexts[k], Q, lambd, batch_size=batch_size)
                for k in range(K)
            ]
        )
        k_step_values = discounted_costs + gamma_powers[:, None] * bootstraps
        bootstrap_steps = np.argmin(k_step_values, axis=0) + 1
        all_bootstrap_steps.append(bootstrap_steps)
        Q_next = np.min(k_step_values, axis=0)

        if np.max(np.abs(Q_next - Q)) < tol:
            avg_bootstrap, median_bootstrap = _bootstrap_step_stats(all_bootstrap_steps)
            return Q_next, avg_bootstrap, median_bootstrap, it

        Q = Q_next

    avg_bootstrap, median_bootstrap = _bootstrap_step_stats(all_bootstrap_steps)
    return Q, avg_bootstrap, median_bootstrap, max_iter


def _bootstrap_step_stats(all_bootstrap_steps: List[np.ndarray]) -> Tuple[float, float]:
    if not all_bootstrap_steps:
        return 1.0, 1.0

    bootstrap_steps = np.concatenate(all_bootstrap_steps)
    return float(np.mean(bootstrap_steps)), float(np.median(bootstrap_steps))



def solve_optimization_problem(
    D: List[Tuple],
    e_min: float = 0,
    lambda_min: float = 1,
    verbose: bool = False,
    w_eps: float = 1.0,
    w_lambda: float = 1.0,
    gamma: float = 0.9,
):
    """
    Based on (xi, ui, ci, xi') transitions in D, solves the optimization problem
    to find a nonparametric Q function that satisfies the Bellman inequalities.
    """
    import gurobipy as gp

    n = len(D)

    x = np.array([d[0] for d in D])
    x_next = np.array([d[3] for d in D])
    costs = np.array([d[2] for d in D])

    dist = np.abs(x.reshape(-1, 1) - x_next.reshape(1, -1))

    model = gp.Model("toy")
    objective = gp.LinExpr()

    q = model.addVars(n, lb=0, name="Q")
    J_next = model.addVars(n, lb=0, name="T")
    e = e_min
    lambd = lambda_min

    objective.add(gp.quicksum(q[i] for i in range(n)))
    objective.add(-w_eps * e - w_lambda * lambd)

    for i in range(n):
        y = model.addVars(n, lb=0, name=f"y_{i}")
        for k in range(n):
            model.addConstr(y[k] == q[k] + lambd * dist[k, i])

        model.addGenConstrMin(J_next[i], [y[k] for k in range(n)], name=f"min_constr_{i}")
        model.addConstr(J_next[i] <= (q[i] - costs[i] - e) / gamma)

    model.setObjective(objective, gp.GRB.MINIMIZE)

    model.setParam("OutputFlag", int(verbose))
    model.setParam("NonConvex", 2)
    model.setParam("MIPGap", 1e-3)

    model.optimize()

    q_val = np.array([var.X for var in q.values()])

    return {
        "model": model,
        "q": q_val,
        "J_next": np.array([var.X for var in J_next.values()]),
        "epsilon": e,
        "lambda": lambd,
        "dist": dist,
    }
