from itertools import groupby

import scipy
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Arrow
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from mimic_alpha import colorAlpha_to_rgb as color_alpha

import numpy as np


DEFAULT_COLORMAP = {
    "Overall Uncertainty": "#e6194B",
    "FirstDiffZeroPerformanceConvergence": "#3cb44b",
    "Performance Convergence": "#ffe119",
    "Max Confidence": "#4363d8",
    "VM": "#f58231",
    "SecondDiffZeroPerformanceConvergence": "#911eb4",
    "FirstDiffZeroStabilizingPredictions-alpha1": "#42d4f4",
    "SC_entropy_mcs": "#f032e6",
    "FirstDiffMinOverallUncertainty": "#bfef45",
    "GOAL": "#fabed4",
    "SC_oracle_acc": "#469990",
    "SecondDiffZeroOverallUncertainty": "#dcbeff",
    "EVM": "#9A6324",
    "SSNCut": "#fffac8",
    "Stabilizing Predictions": "#800000",
    "Uncertainty Convergence": "#aaffc3",
    "FirstDiffZeroOverallUncertainty": "#808000",
    "Classification Change": "#ffd8b1",
    "Contradictory Information": "#000075",
}


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


def pca_error_points(X: np.array, clamp=None, debug: bool = False) -> np.array:
    """
    Given a 2d dataset returns four points representing two lines found from error bars in a PCA decomposition.
    0: x_low, 1: x_high, 2: y_low, 3: y_high
    """
    # Initialize PCA
    pca = PCA(n_components=2, whiten=True, random_state=0)
    # Remove NaN points
    non_nan = X[~np.any(np.isnan(X), axis=1)]
    # Fit PCA
    try:
        pca.fit(non_nan)
    except ValueError as e:
        print("WARN: Initial PCA fit failed")
        print(non_nan.shape)
        raise e
    # Transform points into PCA
    X_t = pca.transform(non_nan)
    # Calculate error bars in PCA coordinates
    x_err_pca = [
        np.percentile(
            X_t[
                0:,
            ],
            2.5,
        ),
        np.percentile(
            X_t[
                0:,
            ],
            97.5,
        ),
    ]
    y_err_pca = [
        np.percentile(
            X_t[
                1:,
            ],
            2.5,
        ),
        np.percentile(
            X_t[
                1:,
            ],
            97.5,
        ),
    ]
    # X Error
    x_l = [
        np.mean(
            X_t[
                0:,
            ]
        )
        + x_err_pca[0],
        np.mean(
            X_t[
                1:,
            ]
        ),
    ]
    x_h = [
        np.mean(
            X_t[
                0:,
            ]
        )
        + x_err_pca[1],
        np.mean(
            X_t[
                1:,
            ]
        ),
    ]
    # Y Error
    y_l = [
        np.mean(
            X_t[
                0:,
            ]
        ),
        np.mean(
            X_t[
                1:,
            ]
        )
        + y_err_pca[0],
    ]
    y_h = [
        np.mean(
            X_t[
                0:,
            ]
        ),
        np.mean(
            X_t[
                1:,
            ]
        )
        + y_err_pca[1],
    ]

    x_l_t, x_h_t, y_l_t, y_h_t = pca.inverse_transform([x_l, x_h, y_l, y_h])

    if debug:
        print("pca", x_l, x_h, y_l, y_h)
        print("err_pca", x_err_pca, y_err_pca)
        print("tra", x_l_t, x_h_t, y_l_t, y_h_t)

    if clamp:
        x_l_t[0], x_h_t[0], y_l_t[0], y_h_t[0] = np.clip(
            [x_l_t[0], x_h_t[0], y_l_t[0], y_h_t[0]], *clamp[0]
        )
        x_l_t[1], x_h_t[1], y_l_t[1], y_h_t[1] = np.clip(
            [x_l_t[1], x_h_t[1], y_l_t[1], y_h_t[1]], *clamp[1]
        )

    return x_l_t, x_h_t, y_l_t, y_h_t


def plot_paraeto_hull(
    results,
    ylims=None,
    hull_alpha=0.3,
    rows=3,
    cols=3,
    figsize=(15, 15),
    dpi=300,
    hull=True,
    error="percentile",
    colors=None,
    marker_size=30,
    legend_loc="upper left",
    legend_cols=1,
    legend_bbox=(1.05, 2.3),
):
    datasets = list(results.keys())
    # we care about the order (consistent markers & legend)
    criteria = []
    for r in results.values():
        for x in r.keys():
            if x not in criteria:
                criteria.append(x)
    # criteria = list({x for r in results.values() for x in r.keys()})
    if colors is None:
        colors = DEFAULT_COLORMAP
    # katmap = sns.color_palette("gist_ncar", 19)
    markers = [
        "s",
        "v",
        "^",
        "*",
        "D",
        "P",
        "o",
        "<",
        ">",
        "X",
        "s",
        "v",
        "^",
        "*",
        "D",
        "P",
        "o",
        "<",
        ">",
        "x",
        "s",
        "v",
        "^",
        "*",
        "D",
        "P",
        "o",
        "<",
        ">",
        "X",
        "s",
        "v",
        "^",
        "*",
        "D",
        "P",
        "o",
        "<",
        ">",
        "x",
    ]

    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)
    for ds in range(len(datasets)):
        r = int(ds) // cols  # div
        c = int(ds) % cols  # mod

        res_name = results[datasets[ds]]
        res_name_f = np.empty((0, 2))
        sc_name = np.empty(0).astype("int")

        for sc, res in res_name.items():
            res_name_f = np.array(
                np.concatenate(
                    (res_name_f, np.array(res)[:, 0:2]),
                ),
                dtype="float",
            )
            sc_name = np.append(sc_name, np.array([criteria.index(sc)] * len(res)))

        pareto_front = res_name_f[_get_frontier(res_name_f) == 1]
        pareto_front = pareto_front[pareto_front[:, 0].argsort()]

        ls = []

        ax[r, c].plot(
            pareto_front[:, 0],
            pareto_front[:, 1],
            c="#777777",
            ls="--",
            zorder=1,
            label="Pareto frontier" if ds == 0 else None,
        )  # Kat
        ax[r, c].grid(color=color_alpha(["black"], 0.2)[0])  # Kat
        ax[r, c].ticklabel_format(style="plain")  # Kat

        if ylims:
            ax[r, c].set_ylim(*ylims)

        for i in range(max(sc_name) + 1):
            # c = '#E0A33D' if i==6 else 'black'
            col = [colors[criteria[i]]]
            points = res_name_f[sc_name == i]
            points = points[~np.isnan(points).any(axis=1)]

            # Plot convex hull around points
            if len(points) >= 3 and hull:
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
                    ax[r, c].fill(x_hull, y_hull, alpha=hull_alpha, c=col[0])
                else:
                    ax[r, c].plot(points[:, 0], points[:, 1], c=col[0], linewidth=1)

            x = res_name_f[sc_name == i, 0]
            y = res_name_f[sc_name == i, 1]
            non_nan = res_name_f[sc_name == i][
                ~np.any(np.isnan(res_name_f[sc_name == i]), axis=1)
            ]
            if error == "percentile":
                xerr = np.expand_dims(
                    np.array(
                        [
                            np.nanmean(x) - np.nanpercentile(x, 2.5),
                            np.nanpercentile(x, 97.5) - np.nanmean(x),
                        ]
                    ),
                    axis=1,
                )
                yerr = np.expand_dims(
                    np.array(
                        [
                            np.nanmean(y) - np.nanpercentile(y, 2.5),
                            np.nanpercentile(y, 97.5) - np.nanmean(y),
                        ]
                    ),
                    axis=1,
                )
            elif error == "std":
                xerr = scipy.stats.sem(x, nan_policy="omit")
                yerr = scipy.stats.sem(y, nan_policy="omit")
            else:
                xerr = None
                yerr = None

            if error != "pca" or points.shape[0] == 0:
                l = ax[r, c].errorbar(
                    x=np.nanmean(x),
                    y=np.nanmean(y),
                    xerr=xerr,
                    yerr=yerr,
                    c=col[0],
                    # markersize=s,
                    marker=markers[i],
                    zorder=3,
                    label=criteria[i],
                    markeredgewidth=1,
                    markeredgecolor="black",
                    ls="",
                )
            elif points.shape[0] > 0:
                # Plot mean & vectors
                l = ax[r, c].scatter(
                    x=np.nanmean(x),
                    y=np.nanmean(y),
                    s=marker_size,
                    c=col,
                    marker=markers[i],
                    zorder=3,
                    label=criteria[i],
                    linewidths=1,
                    edgecolors="black",
                )
                # X error
                if non_nan.shape[0] >= 2:
                    x_l_t, x_h_t, y_l_t, y_h_t = pca_error_points(
                        res_name_f[sc_name == i], debug=False
                    )

                    ax[r, c].plot(
                        [x_l_t[0], x_h_t[0]],
                        [x_l_t[1], x_h_t[1]],
                        color=color_alpha([col[0]], 0.7)[0],
                        # alpha=0.7
                    )
                    ax[r, c].plot(
                        [y_l_t[0], y_h_t[0]],
                        [y_l_t[1], y_h_t[1]],
                        color=color_alpha([col[0]], 0.7)[0],
                        # alpha=0.7
                    )

            ls.append(l)

        if r == rows - 1:
            ax[r, c].set_xlabel("Instances")
        if c == 0:
            ax[r, c].set_ylabel("Accuracy")

        ax[r, c].set_title(datasets[ds].rstrip("-58509"))
        box = ax[r, c].get_position()
        ax[r, c].set_position(
            [box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9]
        )

        # grey area under pareto front
        py = pareto_front[:, 1][np.logical_not(np.isnan(pareto_front[:, 1]))]  # Kat
        ax[r, c].fill_between(
            np.append(
                pareto_front[:, 0][np.logical_not(np.isnan(pareto_front[:, 0]))],
                [ax[r, c].get_xlim()[1]],
            ),
            ax[r, c].get_ylim()[0],
            np.append(py, [py[-1]]),
            facecolor=color_alpha(["#777777"], alpha=0.1)[0],
            # alpha=0.1,
            edgecolor="None",
        )  # Kat

    plt.legend(
        ls,
        [l.get_label() for l in ls],
        bbox_to_anchor=legend_bbox,
        loc=legend_loc,
        ncol=legend_cols,
        frameon=False,
    )
    return fig, ax
