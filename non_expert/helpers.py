from __future__ import annotations

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


def J_upper_bound(y: np.ndarray, x_d: np.ndarray, q: np.ndarray, lambd):
    """Computes J_ub(y) = min_k { Q(xk) + lambda * dist(xk, y) }."""
    assert q.ndim == 1, "q should be a 1D array"
    if hasattr(lambd, "X"):
        lambd = lambd.X
    if y.ndim == 0:
        y = np.array([y])
    dists = np.abs(x_d[:, None] - y[None, :])
    return np.min(q[:, None] + lambd * dists, axis=0)


def Q_upper_bound(y: np.ndarray, u: np.ndarray, x_d: np.ndarray, u_d: np.ndarray, q: np.ndarray, lambd):
    """Computes Q_ub(y, u) = min_k { Q(xk, uk) + lambda * dist(xk, y) + lambda * dist(uk, u) }."""
    assert q.ndim == 1, "q should be a 1D array"
    if hasattr(lambd, "X"):
        lambd = lambd.X
    if y.ndim == 0:
        y = np.array([y])
    dists_x = np.abs(x_d[:, None] - y[None, :])
    dists_u = np.abs(u_d[:, None] - u[None, :])
    return np.min(q[:, None] + lambd * dists_x + lambd * dists_u, axis=0)


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
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a.reshape(a.shape[0], -1)


def find_fixed_point_v2(
    D: List[Tuple],
    lambd: float,
    gamma: float,
    tol: float = 1e-3,
    max_iter: int = 1000,
    warm_Q: Optional[np.ndarray] = None,
):
    """
    Vectorized version of the above.
    Finds the fixed point of the Bellman operator T^Qub:

    (TQ)i = ci + gamma * min_k { Q(xk) + lambda * dist(xk, xi') }
    """
    assert len(D) > 0 and lambd > 0 and 0 < gamma < 1, "Invalid inputs"

    x, _, c, x_next = zip(*D)

    x = as_rows(x)
    x_next = as_rows(x_next)
    c = np.asarray(c, dtype=float).reshape(-1)

    assert x.shape == x_next.shape, "x and x_next must have matching shapes"
    assert x.shape[0] == c.shape[0], "costs must match number of samples"

    dist = np.linalg.norm(x[:, None, :] - x_next[None, :, :], axis=-1)

    Q = np.zeros(x.shape[0]) if warm_Q is None else np.concatenate([warm_Q, np.zeros(len(D) - len(warm_Q))])
    for it in range(max_iter):
        Q_next = c + gamma * np.min(Q[:, None] + lambd * dist, axis=0)

        if np.max(np.abs(Q_next - Q)) < tol:
            print(f"Value Iteration Converged after {it} steps.")
            return Q_next

        Q = Q_next

    return Q


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
