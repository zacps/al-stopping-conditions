import os
import sys
import math
import pickle
import socket
from time import monotonic
from typing import Dict, Any, Callable
import json
import math
from itertools import groupby
import glob
from pprint import pprint, pformat

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from IPython.core.display import HTML, display
except ModuleNotFoundError:
    pass
from libutil import ProgressParallel, out_dir
from joblib import delayed
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import metrics as skmetrics
from sklearn.utils import check_random_state
from sklearn.svm import SVC
from tabulate import tabulate

from traceback_with_variables import activate_by_import

import libdatasets
from libutil import Metrics, average, Notifier, n_cpus
from libstop import first_acc, no_ahead_tvregdiff
from libplot import align_yaxis
from libactive import active_split, MyActiveLearner
from libstore import CompressedStore
from libconfig import Config, Configurations


DEFAULT_METRICS = [
    accuracy_score,
    f1_score,
    roc_auc_score,
    "time",
    "time_total",
    "time_ee",
    "uncertainty_average",
    "uncertainty_min",
    "uncertainty_max",
    "uncertainty_variance",
    "uncertainty_average_selected",
    "uncertainty_min_selected",
    "uncertainty_max_selected",
    "uncertainty_variance_selected",
    "entropy_max",
    "n_support",
    "contradictory_information",
    "expected_error",
    "expected_error_min",
    "expected_error_max",
    "expected_error_average",
    "expected_error_variance",
]


def run(
    matrix,
    force_cache=False,
    force_run=False,
    backend="loky",
    abort=True,
    workers=None,
    metrics=None,
    fragment_id=None,
    fragment_length=1,
    fragment_run_start=None,
    fragment_run_end=None,
    dry_run=False,
):
    print(sys.argv)

    configurations = Configurations(matrix)
    start = monotonic()

    # For NeSI
    if fragment_id is not None:
        configurations.configurations = configurations.configurations[
            fragment_id : fragment_id + fragment_length
        ]

    # Monomorphise parametric meta parameters
    for config in configurations:
        for k, v in config.meta.items():
            if isinstance(v, dict):
                config.meta[k] = v.get(config.dataset_name, v["*"])

    # Exit if this is a dry run
    if dry_run:
        print("Exiting due to dry run:")
        pprint(configurations)

        if fragment_run_end is not None:
            runs = list(range(fragment_run_start, fragment_run_end + 1))
        else:
            runs = [fragment_run_start]

        print(f"Runs: {runs}")

        return

    # Detect number of available CPUs
    if workers is None:
        workers = n_cpus()

    # Calculate the number of repeated runs that were asked for
    if fragment_run_start is not None:
        n_runs = (
            (fragment_run_end - fragment_run_start + 1)
            if fragment_run_end is not None
            else 1
        )
    else:
        n_runs = configurations.meta["n_runs"]

    notifier = Notifier(
        "https://discord.com/api/webhooks/809248326485934080/aIHL726wKxk42YpDI_GtjsqfAWuFplO3QrXoza1r55XRT9-Ao9Rt8sBtexZ-WXSPCtsv"
    )

    try:
        results = ProgressParallel(
            n_jobs=math.ceil(workers / n_runs),
            total=len(configurations),
            desc=f"Experiment",
            leave=False,
            backend=backend,
        )(
            delayed(__run_inner)(
                config,
                force_cache=force_cache,
                force_run=force_run,
                abort=abort,
                metrics_measures=metrics,
                workers=workers,
                fragment_run_start=fragment_run_start,
                fragment_run_end=fragment_run_end,
            )
            for config in configurations
        )
        if configurations.meta["ret_classifiers"]:
            # Figure out what runs we care about
            if fragment_run_start is not None:
                if fragment_run_end is not None:
                    runs = list(range(fragment_run_start, fragment_run_end + 1))
                else:
                    runs = [fragment_run_start]
            else:
                runs = range(config.meta["n_runs"])
            # Retrieve classifiers
            for i, config in enumerate(configurations):
                results[i] = (results[i], [__read_classifiers(config, j) for j in runs])

    except Exception as e:
        duration = monotonic() - start

        notifier.error(configurations, duration, e)
        raise e

    duration = monotonic() - start

    if duration > 10 * 60 or fragment_id is not None:
        notifier.completion(configurations, duration)

    return results


def plot(
    results,
    plot_robustness=False,
    key=None,
    series=None,
    title=None,
    ret=False,
    sort=True,
    figsize=(18, 4),
    extra=0,
    scale="linear",
    hlines=None,
):
    if key is None:
        key = lambda config_result: (
            config_result[0].dataset_name,
            config_result[0].dataset_mutator_name,
            getattr(config_result[0], "model_name", None),
        )
    if series is None:
        series = lambda config: config.method_name
    if title is None:
        title = lambda config: f"{config.dataset_name} {config.model_name}"
    if sort:
        results = sorted(results, key=key)
    groups = groupby(results, key)
    figaxes = []
    for k, group in groups:
        fig, axes = plt.subplots(
            1, (4 if plot_robustness else 3) + extra, figsize=figsize
        )
        figaxes.append((fig, axes))

        for config, result in group:
            if isinstance(result, list):
                if isinstance(result[0], Metrics):
                    result = result[0].average2(result[1:])
                else:
                    result = average(result[0], result[1:])
            for i, ax in enumerate(
                axes.flatten()[: -extra if extra != 0 else len(axes.flatten())]
            ):
                try:
                    i_stderr = result.columns.get_loc("accuracy_score_stderr")
                    has_stderr = True
                except KeyError:
                    has_stderr = False
                if len(result["x"] > 100):
                    ax.plot(
                        result["x"],
                        result.iloc[:, 1 + i],
                        "-",
                        label=f"{series(config)}" if i == 0 else "",
                    )
                    if has_stderr:
                        ax.fill_between(
                            result["x"],
                            result.iloc[:, 1 + i] - result.iloc[:, i_stderr + i],
                            result.iloc[:, 1 + i] + result.iloc[:, i_stderr + i],
                            color="grey",
                            alpha=0.2,
                        )
                else:
                    ax.errorbar(
                        result["x"],
                        result.iloc[:, 1 + i],
                        yerr=result.iloc[:, i_stderr + i] if has_stderr else None,
                        label=f"{series(config)}" if i == 0 else "",
                    )

                if hlines is not None:
                    line = hlines.get((config.dataset_name, config.model_name), None)
                    if line is not None:
                        ax.axhline(line[i])

                ax.set_xlabel("Instances")
                ax.set_ylabel(["Accuracy", "F1", "AUC ROC", "Empirical Robustness"][i])
                ax.set_yscale(scale)
                plt.suptitle(title(config))

        fig.legend()
        fig.tight_layout()
    if ret:
        return figaxes


def table(results, tablefmt="fancy_grid"):
    key = lambda config_result: (
        config_result[0].dataset_name,
        config_result[0].dataset_mutator_name,
        getattr(config_result[0], "model_name", None),
    )
    results = sorted(results, key=key)
    groups = groupby(results, key)

    def max_at(result, has_err, metric):
        try:
            upper = result[metric] + result[f"{metric}_stderr"]
            norm = result[result[metric].ge(result[metric].iloc[-1])].iloc[0].x
            return (
                f"{norm:.0f}"
                + f"±{abs(result[upper.ge(result[metric].iloc[-1])].iloc[0].x-norm):.0f}"
                if has_err
                else ""
            )
        except IndexError:
            return "-"

    def area_under(result, has_err, metric, baseline=1):
        return f"{skmetrics.auc(result['x'], result[metric])/baseline:.4f}" + (
            f"±{(skmetrics.auc(result['x'], result[metric]+result[metric+'_stderr'])-skmetrics.auc(result['x'], result[metric]))/baseline:.0g}"
            if has_err
            else ""
        )

    def baseline(group, metric):
        return next(
            (
                skmetrics.auc(result["x"], result[metric])
                for conf, result in group
                if conf.method_name == "random"
            ),
            1,
        )

    for k, group in groups:
        has_err = not math.isnan(results[0][1].accuracy_score_stderr[0])

        group = list(group)
        print(k[0])
        print(
            tabulate(
                sorted(
                    [
                        [
                            conf.method_name,
                            area_under(
                                result,
                                has_err,
                                "accuracy_score",
                                baseline=baseline(group, "accuracy_score"),
                            ),
                            area_under(
                                result,
                                has_err,
                                "f1_score",
                                baseline=baseline(group, "f1_score"),
                            ),
                            area_under(
                                result,
                                has_err,
                                "roc_auc_score",
                                baseline=baseline(group, "roc_auc_score"),
                            ),
                            max_at(result, has_err, "accuracy_score"),
                            max_at(result, has_err, "roc_auc_score"),
                            result.time.sum() if "time" in result else None,
                        ]
                        for conf, result in group
                    ],
                    key=lambda x: -float(x[1].split("±")[0]),
                ),
                tablefmt=tablefmt,
                headers=[
                    "method",
                    "AUC LAC",
                    "AUC LF1C",
                    "AUC AUC ROC",
                    "Instances to max accuracy",
                    "Instances to max AUC ROC",
                    "Time",
                ],
            )
        )


def __run_inner(
    config,
    force_cache=False,
    force_run=False,
    backend="loky",
    abort=None,
    metrics_measures=None,
    workers=None,
    fragment_run_start=None,
    fragment_run_end=None,
):
    # Default metrics are deprecated
    if metrics_measures is None:
        meatrics_measures = DEFAULT_METRICS

    # Figure out what runs we care about
    if fragment_run_start is not None:
        if fragment_run_end is not None:
            runs = list(range(fragment_run_start, fragment_run_end + 1))
        else:
            runs = [fragment_run_start]
    else:
        runs = list(range(config.meta["n_runs"]))

    # Set the number of worker threads
    if workers is None:
        workers = n_cpus()

    # Attempt to restore a cached result
    if not force_run:
        try:
            cached_config, metrics = __read_result(
                f"{out_dir()}{os.path.sep}{config.serialize()}.csv", config, runs=runs
            )

            return (cached_config, metrics)
        except (FileNotFoundError, EOFError, pd.errors.EmptyDataError):
            pass

    # If we didn't find a cached result and force_cache is set raise an error
    if force_cache:
        raise Exception(
            f"Cache file '{out_dir()}{os.path.sep}{config.serialize()}.csv' not found"
        )

    try:
        # Seed a random state generator. This seed is constant between methods/datasets/models so comparisons can be made with fewer runs.
        # It is however *variant* with each run.
        random_state = [check_random_state(i) for i in runs]

        metrics = ProgressParallel(
            n_jobs=min(len(runs), workers),
            total=len(runs),
            desc=f"Run",
            leave=False,
            backend=backend,
        )(
            delayed(
                lambda dataset, method, i, random_state: MyActiveLearner(
                    # It's important that the split is re-randomised per run.
                    *active_split(
                        *dataset,
                        labeled_size=config.meta.get("labelled_size", 0.1),
                        test_size=config.meta.get("test_size", 0.5),
                        ensure_y=config.meta.get("ensure_y", False),
                        random_state=random_state,
                        mutator=config.dataset_mutator,
                        config_str=config.serialize(),
                        i=i,
                    ),
                    method,
                    config,
                    metrics=metrics_measures,
                    i=i,
                ).active_learn2()
            )(config.dataset(), config.method, i, random_state[idx])
            for idx, i in enumerate(runs)
        )

    except Exception as e:
        if abort:
            raise e
        print("WARN: Experiment failed, continuing anyway")
        return (config, None)

    if config.meta.get("aggregate", True):
        metrics = metrics[0].average2(metrics[1:])

    __write_result(config, metrics, runs)
    for i in runs:
        try:
            os.remove(
                f"{out_dir()}{os.path.sep}runs{os.path.sep}{config.serialize()}_{i}.csv"
            )
        except FileNotFoundError:
            pass

    return (config, metrics)


def plot_stop(
    plots, classifiers, stop_conditions, stop_results, scale="linear", figsize=(26, 4)
):
    figaxes = plot(plots, ret=True, sort=False, extra=2, scale=scale, figsize=figsize)
    for i, (fig, ax) in enumerate(figaxes):
        clfs = classifiers[i]
        metrics = plots[i][1]

        for j, clfs_ in enumerate(clfs):
            if len(clfs_) < 100:
                raise Exception(
                    f"short classifier file: {plots[i][0].serialize()}_{j}.zip\nIt has length {len(clfs_)} when it should have length 100"
                )

        if plots[i][0].dataset_mutator_name != "none":
            scores = __get_passive_scores(plots[i][0], range(len(plots[i][1])))
            for ax, score in zip(ax, scores):
                ax.axhline(score, color="tab:gray", ls="--")

        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        accs = [first_acc(clfs_)[1] for clfs_ in clfs]
        accx = first_acc(clfs[0])[0]

        acc_median = np.median(accs, axis=0)
        acc_stderr = np.std(accs, axis=0)

        ax[-1].plot(metrics[0].x[: acc_median.shape[0]], acc_median)
        ax[-1].set_title("First classifier accuracy")

        ax1 = ax[-1].twinx()
        ax1.axhline(0, ls="--", color="grey", alpha=0.8)
        ax1.plot(
            metrics[0].x[: acc_median.shape[0]],
            no_ahead_tvregdiff(acc_median, 1, 1e-1, plotflag=False, diagflag=False),
            ls="--",
        )

        if "expected_error_min" in metrics[0]:
            ax[-2].plot(metrics[0].x, metrics[0].expected_error_min, ls="--")
            ax[-2].set_title("expected_error_min")

            ax2 = ax[-2].twinx()
            ax2.axhline(0, ls="--", color="grey", alpha=0.8)
            ee_first = no_ahead_tvregdiff(
                metrics[0].expected_error_min[1:],
                1,
                1e2,
                plotflag=False,
                diagflag=False,
            )
            ee_second = no_ahead_tvregdiff(
                ee_first[2:], 1, 15, plotflag=False, diagflag=False
            )
            ax2.plot(
                metrics[0].x[1:], ee_first / np.max(np.abs(ee_first[2:])), label="1st"
            )
            ax2.plot(
                metrics[0].x[3:], ee_second / np.max(np.abs(ee_second[2:])), label="2nd"
            )
            ax2.legend()

            align_yaxis(ax[-2], ax2)

        for ii, a in enumerate(ax):
            for iii, (name, cond) in enumerate(stop_conditions.items()):
                stops = stop_results[plots[i][0].dataset_name][name]
                for iiii, stop in enumerate(stops):
                    if stop[0] is not None:
                        # print(stop)
                        a.axvline(
                            stop[0],
                            label=name if ii == 0 and iiii == 0 else None,
                            color=colors[(iii + 1) % len(colors)],
                        )
                        break

        fig.legend()
        fig.tight_layout()


def __get_passive_scores(conf, runs):
    """
    Get the performance scores that would be obtained by a passive classifier trained on all the
    perfect data.
    """
    fname = f"{out_dir()}{os.path.sep}passive{os.path.sep}{conf.dataset_name}_{conf.model_name}.csv"
    try:
        with open(fname, "rb") as f:
            results = pickle.load(f)
            if all([run in last.keys() for run in runs]):
                return [
                    [
                        np.min([result[i] for result in results]),
                        np.mean([result[i] for result in results]),
                        np.max([result[i] for result in results]),
                    ]
                    for i in range(3)
                ]
    except FileNotFoundError:
        results = {}

    assert conf.model_name == "svm-linear"

    X, y = getattr(libdatasets, conf.dataset_name)(None)

    for i in runs:
        if i in results.keys():
            continue
        _, X_unlabelled, y_labelled, y_oracle, X_test, y_test = active_split(
            X,
            y,
            labeled_size=conf.meta["labelled_size"],
            test_size=conf.meta["test_size"],
            random_state=check_random_state(i),
            ensure_y=conf.meta["ensure_y"],
        )
        clf = SVC(probability=True, kernel="linear")
        clf.fit(X_unlabelled, y_oracle)
        predicted = clf.predict(X_test)
        predict_proba = clf.predict_proba(X_test)
        unique_labels = np.unique(y_labelled)

        if len(unique_labels) > 2 or len(unique_labels.shape[0]) > 1:
            roc_auc = roc_auc_score(y_test, predict_proba, multi_class="ovr")
        else:
            roc_auc = roc_auc_score(y_test, predict_proba[:, 1])

        results[i] = [
            accuracy_score(y_test, predicted),
            f1_score(
                y_test,
                predicted,
                average="micro" if len(unique_labels) > 2 else "binary",
                pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1,
            ),
            roc_auc,
        ]
        with open(fname, "wb") as f:
            pickle.dump(f, results)

    return [
        [
            np.min([result[i] for result in results]),
            np.mean([result[i] for result in results]),
            np.max([result[i] for result in results]),
        ]
        for i in range(3)
    ]


def __write_result(config, result, runs):
    if isinstance(result, list):
        for r, run in zip(result, runs):
            file = f"{out_dir()}{os.path.sep}{config.serialize()}_{run}.csv"
            with open(file, "w") as f:
                json.dump(config.json(), f)
                f.write("\n")
                r.frame.to_csv(f)
    else:
        file = f"{out_dir()}{os.path.sep}{config.serialize()}.csv"
        with open(file, "w") as f:
            json.dump(config.json(), f)
            f.write("\n")
            result.to_csv(f)


def __read_classifiers(config, i=None):
    if False and config.meta["stop_function"][0] != "len1000":
        c_str = config.serialize().replace(config.meta["stop_function"][0], "len1000")
    else:
        c_str = config.serialize()

    pfile = (
        f"{out_dir()}{os.path.sep}classifiers{os.path.sep}{config.serialize()}.pickle"
    )
    zfile = f"{out_dir()}{os.path.sep}classifiers{os.path.sep}{c_str}_{i}.zip"
    try:
        with open(pfile, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return CompressedStore(zfile, read=True)


def __read_result(file, config, runs=None):
    if config.meta.get("aggregate", True):
        with open(file, "r") as f:
            cached_config = Config(
                **{"model_name": "svm-linear", **json.loads(f.readline())}
            )
            result = pd.read_csv(f, index_col=0)
        return (cached_config, result)
    else:
        results = []
        for name in [
            f"{out_dir()}{os.path.sep}{config.serialize()}_{i}.csv" for i in runs
        ]:
            with open(name, "r") as f:
                cached_config = Config(
                    **{"model_name": "svm-linear", **json.loads(f.readline())}
                )
                results.append(pd.read_csv(f, index_col=0))
        # make the run numbers available
        cached_config.runs = runs
        return cached_config, results


def __progress_hack():
    # Annoying hack so that the progressbars disapear as they're supposed to
    try:
        display(
            HTML(
                """
        <style>
        .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
        padding: 0;
        border: 0;
        }
        .p-Widget.jp-RenderedText.jp-OutputArea-output pre:empty {
        display: none;
        }
        </style>
        """
            )
        )
    except NameError:
        pass
