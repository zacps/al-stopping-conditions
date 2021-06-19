from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Arrow
import matplotlib.patches as patches

import numpy as np


def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def make_meshgrid_bounds(x_min, x_max, y_min, y_max, h=0.02):
    """Create a mesh of points to plot in

    Returns
    -------
    xx, yy : ndarray
    """

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    if type(Z) is not np.ndarray:
        Z = Z.tondarray()
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_poison_contours(ax, attack, xx, yy, **params):
    Z = np.vectorize(attack.objective_function, signature="(2)->()")(
        np.concatenate((xx.reshape((xx.size, 1)), yy.reshape((yy.size, 1))), axis=1)
    )
    if type(Z) is not np.ndarray:
        Z = Z.tondarray()
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, levels=20, **params)
    return out


def plot_classification(ax, clf, X, Y, X_all):
    """
    Plot the results of a classifier
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X_all[:, 0], X_all[:, 1])

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.7)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())


def plot_poison(
    clf,
    X_labelled,
    y_labelled,
    X_unlabelled,
    y_unlabelled,
    X_test,
    y_test,
    attack,
    attack_points,
    start_points,
    start_points_y,
    ax=None,
):
    """
    Plot the results of a poison attack
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    if X_unlabelled is not None:
        X_all = np.concatenate((X_labelled, X_unlabelled, X_test), axis=0)
    else:
        X_all = np.concatenate((X_labelled, X_test), axis=0)

    if X_unlabelled is not None:
        ax.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], s=10)

    X0, X1 = X_labelled[:, 0], X_labelled[:, 1]
    xx, yy = make_meshgrid(X_all[:, 0], X_all[:, 1])

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_labelled, cmap=plt.cm.coolwarm, s=25, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    if len(attack_points.shape) > 2:
        attack_points = attack_points[0]

    for start, end, start_label in zip(start_points, attack_points, start_points_y):
        color = "#526dde" if start_label else "#d14138"
        ax.arrow(
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
            shape="full",
            color="black",
            length_includes_head=True,
            zorder=1,
            head_length=0.15,
            head_width=0.1,
        )


def c_plot_poison(
    X_labelled,
    y_labelled,
    attack,
    lb,
    ub,
    attack_points,
    start_points,
    start_points_y,
    x_seq,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))

    X0, X1 = X_labelled[:, 0], X_labelled[:, 1]
    xx, yy = make_meshgrid_bounds(lb[0], ub[0], lb[1], ub[1], h=0.2)

    if attack is not None:
        plot_poison_contours(ax, attack, xx, yy, cmap="jet", alpha=0.8)

    ax.scatter(X0, X1, c=y_labelled, cmap=plt.cm.coolwarm, s=25, edgecolors="k")
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())

    if attack_points is not None and start_points is not None and x_seq is None:
        if len(attack_points.shape) > 2:
            attack_points = attack_points[0]

        for start, end, start_label in zip(start_points, attack_points, start_points_y):
            color = "#526dde" if start_label else "#d14138"
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                shape="full",
                color="black",
                length_includes_head=True,
                zorder=1,
                head_length=0.15,
                head_width=0.1,
            )

    if x_seq is not None:
        x_seq = x_seq.tondarray()
        x_seq = remove_consecutive_dups(x_seq)
        path = [(Path.MOVETO, x_seq[0]), *[(Path.LINETO, x) for x in x_seq[1:-1]]]

        codes, verts = zip(*path)
        string_path = Path(verts, codes)
        patch = patches.PathPatch(string_path, facecolor="none", lw=2)

        ax.add_patch(patch)

        # arrow = Arrow(x_seq[-2][0], x_seq[-2][1], x_seq[-1][0]-x_seq[-2][0], x_seq[-1][1]-x_seq[-2][1])
        # print("Added arrow")
        # ax.add_patch(arrow)
        if len(x_seq) >= 2:
            ax.arrow(
                x_seq[-2][0],
                x_seq[-2][1],
                x_seq[-1][0] - x_seq[-2][0],
                x_seq[-1][1] - x_seq[-2][1],
                shape="full",
                color="black",
                length_includes_head=True,
                zorder=10,
                head_length=0.05,
                head_width=0.05,
            )


# See https://stackoverflow.com/questions/5738901/removing-elements-that-have-consecutive-duplicates
def remove_consecutive_dups(array):
    return np.array([x[0] for x in groupby(array.tolist())])


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = np.array([ax1, ax2])
    extrema = np.array([ax.get_ylim() for ax in axes])
    tops = extrema[:, 1] / (extrema[:, 1] - extrema[:, 0])
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [a[::-1] for a in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    extrema[0, 1] = extrema[0, 0] + tot_span * (extrema[0, 1] - extrema[0, 0])
    extrema[1, 0] = extrema[1, 1] + tot_span * (extrema[1, 0] - extrema[1, 1])
    [axes[i].set_ylim(*extrema[i]) for i in range(2)]


# ----------------------------------------------------------------------------------------------
# Paraeto plots
# ----------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import seaborn as sns


def _get_frontier(res):
    "find pareto-optima"
    tmp = np.column_stack((res[:, 0], 1 - res[:, 1]))
    frontier = np.ones(len(tmp))
    for i in range(len(tmp)):
        for r2 in tmp:
            if all(r2 <= tmp[i]) and any(r2 < tmp[i]):
                frontier[i] = 0  # res[i] is dominated by r2!
                break
    return frontier.astype(int)


def plot_paraeto_hull(results):
    datasets = list(results.keys())
    criteria = list(results[datasets[0]].keys())
    katmap = sns.color_palette("gist_ncar", 7)
    markers = ["s", "v", "^", "*", "D", "P", "o"]

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for ds in range(len(datasets)):
        r = int(ds) // 3  # div
        c = int(ds) % 3  # mod

        res_name = results[datasets[ds]]
        res_name_f = np.empty((0, 2))
        sc_name = np.empty(0).astype("int")

        for sc, res in res_name.items():
            # res_anu_avg = np.row_stack((res_anu_avg, np.nanmean(np.array(res, dtype=np.float), axis=0)[0:2]))
            res_name_f = np.concatenate(
                (res_name_f, np.array(res, dtype=np.float)[:, 0:2])
            )
            sc_name = np.append(sc_name, np.array([criteria.index(sc)] * len(res)))

        pareto_front = res_name_f[_get_frontier(res_name_f) == 1]
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]

        ax[r, c].plot(pareto_front[:, 0], pareto_front[:, 1], c="#3D6AE0")
        for i in range(max(sc_name) + 1):
            # c = '#E0A33D' if i==6 else 'black'
            col = [katmap[i]]
            points = res_name_f[sc_name == i]
            points = points[~np.isnan(points).any(axis=1)]

            if len(points) > 0:
                if (
                    len(np.unique(points[:, 0])) > 1
                    and len(np.unique(points[:, 1])) > 1
                ):
                    hull = ConvexHull(points)
                    x_hull = np.append(
                        points[hull.vertices, 0], points[hull.vertices, 0][0]
                    )
                    y_hull = np.append(
                        points[hull.vertices, 1], points[hull.vertices, 1][0]
                    )
                    ax[r, c].fill(x_hull, y_hull, alpha=0.3, c=col[0])
                else:
                    ax[r, c].plot(points[:, 0], points[:, 1], c=col[0], linewidth=1)

            s = 40 if i == 6 else 20
            ax[r, c].scatter(
                res_name_f[sc_name == i, 0],
                res_name_f[sc_name == i, 1],
                c=col,
                s=s,
                marker=markers[i],
                zorder=3,
                label=criteria[i],
            )
        ax[r, c].plot(pareto_front[:, 0], pareto_front[:, 1], c="#3D6AE0")
        if r == 2:
            ax[r, c].set_xlabel("# Instances")
        if c == 0:
            ax[r, c].set_ylabel("Accuracy")
        if ds == 8:
            ax[r, c].legend()
        ax[r, c].set_title(datasets[ds])
    plt.show()
