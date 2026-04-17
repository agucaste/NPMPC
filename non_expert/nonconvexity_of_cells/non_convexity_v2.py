import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ============================================================
# Norms
# ============================================================
def l1_norm(x, c):
    return np.abs(x[..., 0] - c[0]) + np.abs(x[..., 1] - c[1])


def l2_norm(x, c):
    return np.sqrt((x[..., 0] - c[0]) ** 2 + (x[..., 1] - c[1]) ** 2)


def linf_norm(x, c):
    return np.maximum(np.abs(x[..., 0] - c[0]), np.abs(x[..., 1] - c[1]))


# ============================================================
# Plot helper
# ============================================================
def plot_shifted_cell(
    ax,
    norm_fun,
    ci,
    cj,
    delta,
    points,
    title,
    xlim,
    ylim,
    grid_n=900,
    eps=1e-9,
):
    """
    Plot the shifted cell
        C_i^delta = {x : ||x-ci|| <= ||x-cj|| + delta}
    and explicitly visualize:
        interior:   F < -eps
        boundary:  |F| <= eps
        exterior:   F > eps
    where
        F(x) = ||x-ci|| - ||x-cj|| - delta.
    """

    xs = np.linspace(xlim[0], xlim[1], grid_n)
    ys = np.linspace(ylim[0], ylim[1], grid_n)
    X, Y = np.meshgrid(xs, ys)
    XY = np.stack([X, Y], axis=-1)

    Fi = norm_fun(XY, ci)
    Fj = norm_fun(XY, cj)
    F = Fi - Fj - delta

    # Three-way classification
    # -1 = interior
    #  0 = boundary / zero-measure or positive-measure "thick boundary"
    # +1 = exterior
    Z = np.ones_like(F, dtype=int)
    Z[F < -eps] = -1
    Z[np.abs(F) <= eps] = 0

    # Colors: interior / boundary / exterior
    cmap = ListedColormap(["#8ecae6", "#ffb703", "#d9d9d9"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    ax.imshow(
        Z,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
        alpha=0.85,
    )

    # Draw the nominal contour F = 0 as well
    # This is useful, but it may fail to reveal "thick" boundaries by itself.
    ax.contour(X, Y, F, levels=[0], colors="k", linewidths=1.8)

    # Centers
    ax.scatter(*ci, s=90, marker="o", color="black", zorder=5)
    ax.scatter(*cj, s=90, marker="s", color="black", zorder=5)
    ax.annotate(r"$c_i$", ci, xytext=(6, 6), textcoords="offset points", fontsize=11)
    ax.annotate(r"$c_j$", cj, xytext=(6, 6), textcoords="offset points", fontsize=11)

    # Example points
    for label, pt in points.items():
        ax.scatter(pt[0], pt[1], s=70, color="crimson", zorder=6)
        ax.annotate(label, pt, xytext=(6, 6), textcoords="offset points", fontsize=11)

    if "p" in points and "q" in points:
        pq = np.vstack([points["p"], points["q"]])
        ax.plot(pq[:, 0], pq[:, 1], "--", color="crimson", linewidth=1.5)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.25)

    # Print some diagnostics
    frac_boundary = np.mean(np.abs(F) <= eps)
    ax.text(
        0.02,
        0.02,
        rf"$\delta={delta}$" + "\n" + rf"boundary fraction $\approx$ {frac_boundary:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )


# ============================================================
# Counterexamples
# ============================================================

# ----------------------------
# l2 example
# ----------------------------
a = 2.0
delta_l2 = 1.2
T = 2.5
u_T = (delta_l2 / 2.0) * np.sqrt(1.0 + T**2 / (a**2 - (delta_l2 / 2.0) ** 2))

ci_l2 = (-a, 0.0)
cj_l2 = (a, 0.0)
p_l2 = (u_T, T)
q_l2 = (u_T, -T)
m_l2 = (u_T, 0.0)

# ----------------------------
# l1 example
# ----------------------------
delta_l1 = 1.0
ci_l1 = (0.0, 0.0)
cj_l1 = (1.0, 1.0)
p_l1 = (2.0, 0.0)
q_l1 = (0.0, 2.0)
m_l1 = (1.0, 1.0)

# ----------------------------
# linf example
# ----------------------------
delta_linf = 0.5
ci_linf = (-1.0, 0.0)
cj_linf = (1.0, 0.0)
p_linf = (2.0, 3.0)
q_linf = (2.0, -3.0)
m_linf = (2.0, 0.0)


# ============================================================
# Main figure
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)

plot_shifted_cell(
    axes[0],
    l2_norm,
    ci_l2,
    cj_l2,
    delta_l2,
    {"p": p_l2, "q": q_l2, "m": m_l2},
    title=r"$\ell_2$: $\,\|x-c_i\|_2 \leq \|x-c_j\|_2 + \delta$",
    xlim=(-3.5, 3.5),
    ylim=(-3.2, 3.2),
    grid_n=900,
    eps=5e-3,
)

plot_shifted_cell(
    axes[1],
    l1_norm,
    ci_l1,
    cj_l1,
    delta_l1,
    {"p": p_l1, "q": q_l1, "m": m_l1},
    title=r"$\ell_1$: $\,\|x-c_i\|_1 \leq \|x-c_j\|_1 + \delta$",
    xlim=(-1.5, 3.0),
    ylim=(-1.5, 3.0),
    grid_n=900,
    eps=1e-12,
)

plot_shifted_cell(
    axes[2],
    linf_norm,
    ci_linf,
    cj_linf,
    delta_linf,
    {"p": p_linf, "q": q_linf, "m": m_linf},
    title=r"$\ell_\infty$: $\,\|x-c_i\|_\infty \leq \|x-c_j\|_\infty + \delta$",
    xlim=(-3.0, 3.5),
    ylim=(-4.0, 4.0),
    grid_n=900,
    eps=1e-12,
)

fig.suptitle(
    r"Shifted pairwise cells $C_i^\delta=\{x:\|x-c_i\|\leq \|x-c_j\|+\delta\}$"
    "\n"
    r"Blue = interior, orange = boundary region ($|F|\leq \varepsilon$), gray = exterior",
    fontsize=15,
)

plt.show()


# ============================================================
# Optional: isolate just the l1 and linf cases at higher res
# to inspect thick boundaries more clearly
# ============================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)

plot_shifted_cell(
    axes2[0],
    l1_norm,
    ci_l1,
    cj_l1,
    delta_l1,
    {"p": p_l1, "q": q_l1, "m": m_l1},
    title=r"Zoom on $\ell_1$",
    xlim=(-1.5, 3.0),
    ylim=(-1.5, 3.0),
    grid_n=1400,
    eps=1e-12,
)

plot_shifted_cell(
    axes2[1],
    linf_norm,
    ci_linf,
    cj_linf,
    delta_linf,
    {"p": p_linf, "q": q_linf, "m": m_linf},
    title=r"Zoom on $\ell_\infty$",
    xlim=(-3.0, 3.5),
    ylim=(-4.0, 4.0),
    grid_n=1400,
    eps=1e-12,
)

plt.show()