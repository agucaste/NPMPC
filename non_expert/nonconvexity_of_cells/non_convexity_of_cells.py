from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def l1_norm(x, c):
    return np.abs(x[..., 0] - c[0]) + np.abs(x[..., 1] - c[1])


def l2_norm(x, c):
    return np.sqrt((x[..., 0] - c[0]) ** 2 + (x[..., 1] - c[1]) ** 2)


def linf_norm(x, c):
    return np.maximum(np.abs(x[..., 0] - c[0]), np.abs(x[..., 1] - c[1]))


def plot_counterexample(
    ax,
    norm_fun,
    ci,
    cj,
    delta,
    points,
    title,
    xlim,
    ylim,
    grid_n=600,
):
    """
    Plot the set
        S = {x : ||x-ci|| <= ||x-cj|| + delta}
    together with its boundary and annotated points.
    """

    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X, Y], axis=-1)

    fi = norm_fun(XY, ci)
    fj = norm_fun(XY, cj)
    F = fi - fj - delta

    # Cell colors
    # S  : F <= 0
    # Sc : F > 0
    # ax.contourf(
    #     X,
    #     Y,
    #     F,
    #     levels=[-1e6, 0, 1e6],
    #     alpha=0.28,
    # )

    # Boundary F = 0
    # ax.contour(X, Y, F, levels=[0], linewidths=2)
    
    eps = 1e-9
    Z = np.zeros_like(F)

    Z[F < -eps] = -1      # inside
    Z[np.abs(F) <= eps] = 0  # boundary
    Z[F > eps] = 1       # outside

    cmap = ListedColormap([
    "#8ecae6",   # inside
    "#999696", # boundary 
    "#f1ddaa", # outside
    ])

    ax.imshow(Z, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], origin="lower", interpolation="nearest", cmap=cmap)

    # Plot centers
    ax.scatter(*ci, s=70, marker="o")
    ax.scatter(*cj, s=70, marker="s")
    ax.annotate(r"$c_i$", ci, xytext=(6, 6), textcoords="offset points")
    ax.annotate(r"$c_j$", cj, xytext=(6, 6), textcoords="offset points")

    # Plot example points
    for label, pt in points.items():
        ax.scatter(pt[0], pt[1], s=60)
        ax.annotate(label, pt, xytext=(6, 6), textcoords="offset points")

    # Segment joining p and q, to emphasize midpoint argument
    if "p" in points and "q" in points:
        pq = np.vstack([points["p"], points["q"]])
        ax.plot(pq[:, 0], pq[:, 1], "--", linewidth=1.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


# ------------------------------------------------------------
# Counterexample 1: l2
# S = {x : ||x-ci||_2 <= ||x-cj||_2 + delta}
# ci = (-a,0), cj = (a,0), 0 < delta < 2a
# pick p=(u_T,T), q=(u_T,-T), midpoint m=(u_T,0)
# ------------------------------------------------------------
a = 2.0
delta_l2 = 1.2
T = 2.5
u_T = (delta_l2 / 2.0) * np.sqrt(1.0 + T**2 / (a**2 - (delta_l2 / 2.0) ** 2))

ci_l2 = (-a, 0.0)
cj_l2 = (a, 0.0)

p_l2 = (u_T, T)
q_l2 = (u_T, -T)
m_l2 = (u_T, 0.0)

# ------------------------------------------------------------
# Counterexample 2: l1
# ci=(0,0), cj=(1,1), delta in [0,2)
# p=(2,0), q=(0,2), midpoint m=(1,1)
# ------------------------------------------------------------
delta_l1 = 1.0
ci_l1 = (0.0, 0.0)
cj_l1 = (1.0, 1.0)

p_l1 = (2.0, 0.0)
q_l1 = (0.0, 2.0)
m_l1 = (1.0, 1.0)

# ------------------------------------------------------------
# Counterexample 3: l_infinity
# ci=(-1,0), cj=(1,0), delta=1/2
# p=(2,3), q=(2,-3), midpoint m=(2,0)
# ------------------------------------------------------------
delta_linf = 0.5
ci_linf = (-1.0, 0.0)
cj_linf = (1.0, 0.0)

p_linf = (2.0, 3.0)
q_linf = (2.0, -3.0)
m_linf = (2.0, 0.0)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

plot_counterexample(
    axes[0],
    l2_norm,
    ci_l2,
    cj_l2,
    delta_l2,
    {"p": p_l2, "q": q_l2, "m": m_l2},
    title=rf"$\ell_2$ counterexample: $\|x-c_i\|_2 \leq \|x-c_j\|_2 + \delta$,  $\delta={delta_l2}$",
    xlim=(-3.5, 3.5),
    ylim=(-3.2, 3.2),
)

plot_counterexample(
    axes[1],
    l1_norm,
    ci_l1,
    cj_l1,
    delta_l1,
    {"p": p_l1, "q": q_l1, "m": m_l1},
    title=rf"$\ell_1$ counterexample: $\|x-c_i\|_1 \leq \|x-c_j\|_1 + \delta$,  $\delta={delta_l1}$",
    xlim=(-1.5, 3.0),
    ylim=(-1.5, 3.0),
)

plot_counterexample(
    axes[2],
    linf_norm,
    ci_linf,
    cj_linf,
    delta_linf,
    {"p": p_linf, "q": q_linf, "m": m_linf},
    title=rf"$\ell_\infty$ counterexample: $\|x-c_i\|_\infty \leq \|x-c_j\|_\infty + \delta$,  $\delta={delta_linf}$",
    xlim=(-3.0, 3.5),
    ylim=(-4.0, 4.0),
)

# Add a simple shared note
fig.suptitle(
    r"Counterexamples to convexity of $S=\{x:\|x-c_i\|\leq \|x-c_j\|+\delta\}$",
    fontsize=15,
)

plt.savefig('counterexamples_to_convexity.png', dpi=300)

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

delta_l1, delta_l2, delta_linf = [0] * 3

plot_counterexample(
    axes[0],
    l2_norm,
    ci_l2,
    cj_l2,
    delta_l2,
    {"p": p_l2, "q": q_l2, "m": m_l2},
    title=rf"$\ell_2$ example: $\|x-c_i\|_2 \leq \|x-c_j\|_2 + \delta$,  $\delta={delta_l2}$",
    xlim=(-3.5, 3.5),
    ylim=(-3.2, 3.2),
)

plot_counterexample(
    axes[1],
    l1_norm,
    ci_l1,
    cj_l1,
    delta_l1,
    {"p": p_l1, "q": q_l1, "m": m_l1},
    title=rf"$\ell_1$ example: $\|x-c_i\|_1 \leq \|x-c_j\|_1 + \delta$,  $\delta={delta_l1}$",
    xlim=(-1.5, 3.0),
    ylim=(-1.5, 3.0),
)

plot_counterexample(
    axes[2],
    linf_norm,
    ci_linf,
    cj_linf,
    delta_linf,
    {"p": p_linf, "q": q_linf, "m": m_linf},
    title=rf"$\ell_\infty$ example: $\|x-c_i\|_\infty \leq \|x-c_j\|_\infty + \delta$,  $\delta={delta_linf}$",
    xlim=(-3.0, 3.5),
    ylim=(-4.0, 4.0),
)

# Add a simple shared note
fig.suptitle(
    r"Counterexamples to convexity of $S=\{x:\|x-c_i\|\leq \|x-c_j\|+\delta\}$",
    fontsize=15,
)

plt.savefig('counterexamples_of_convexity_d0.png', dpi=300)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)

delta_l1, delta_l2, delta_linf = [0] * 3

ci_l1 = (0, 0)
cj_l1 = (1, 2)
p_l1 = (0, 1)
q_l1 = (2, 0)
m_l1 = (1, .5)

ci_linf = ci_l1
cj_linf = (3, 1)
p_linf = (1, 1)
q_linf = (2, -2)
m_linf = (1.5, -0.5)

# plot_counterexample(
#     axes[0],
#     l2_norm,
#     ci_l2,
#     cj_l2,
#     delta_l2,
#     {"p": p_l2, "q": q_l2, "m": m_l2},
#     title=rf"$\ell_2$ example: $\|x-c_i\|_2 \leq \|x-c_j\|_2 + \delta$,  $\delta={delta_l2}$",
#     xlim=(-3.5, 3.5),
#     ylim=(-3.2, 3.2),
# )

plot_counterexample(
    axes[1],
    l1_norm,
    ci_l1,
    cj_l1,
    delta_l1,
    {},
    # {"p": p_l1, "q": q_l1, "m": m_l1},
    title=rf"$\ell_1$ example: $\|x-c_i\|_1 \leq \|x-c_j\|_1 + \delta$,  $\delta={delta_l1}$",
    xlim=(-1.5, 3.0),
    ylim=(-1.5, 3.0),
)

plot_counterexample(
    axes[2],
    linf_norm,
    ci_linf,
    cj_linf,
    delta_linf,
    {},
    # {"p": p_linf, "q": q_linf, "m": m_linf},
    title=rf"$\ell_\infty$ example: $\|x-c_i\|_\infty \leq \|x-c_j\|_\infty + \delta$,  $\delta={delta_linf}$",
    xlim=(-3.0, 3.5),
    ylim=(-4.0, 4.0),
)

# Add a simple shared note
fig.suptitle(
    r"Counterexamples to convexity of $S=\{x:\|x-c_i\|\leq \|x-c_j\|+\delta\}$",
    fontsize=15,
)

plt.savefig('strict_counterexamples_of_convexity_d0.png', dpi=300)