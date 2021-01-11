import os
import io
import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Callable
from functools import partial
import json
import math
from itertools import groupby

import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
from libutil import ProgressParallel
from joblib import delayed
from libactive import active_split, MyActiveLearner
from libadversarial import random_batch
from modAL import batch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import metrics as skmetrics
from art.metrics import empirical_robustness
from tabulate import tabulate

DEFAULT_MATRIX = {
    # Dataset fetchers should cache if possible
    "datasets": [("syntheic", "generateData_twoPills_2D"), ("car", lambda: "car")],
    "dataset_mutators": {
        "none": (lambda x: x),
    },
    "methods": [
        ("random", partial(random_batch, n_instances=10)),
        ("uncertainty", partial(batch.uncertainty_batch_sampling, n_instances=10)),
    ],
    "meta": {"dataset_size": 1000, "labelled_size": 0.1, "n_runs": 10},
}


@dataclass
class Config:
    dataset_name: str
    method_name: str
    dataset_mutator_name: str
    meta: Dict["str", Any]
    method: Callable = None
    dataset: Callable = None
    dataset_mutator: Callable = None

    def serialize(self):
        meta_str = "__".join([f"{k}={v}" for k, v in self.meta.items()])
        return f"{self.dataset_name}__{self.dataset_mutator_name}__{self.method_name}__{meta_str}"

    def json(self):
        return {
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "dataset_mutator_name": self.dataset_mutator_name,
            "meta": self.meta,
        }


class Configurations:
    def __init__(self, matrix):
        self.configurations = []
        self.meta = matrix["meta"]

        for dataset in matrix["datasets"]:
            for method in matrix["methods"]:
                for dataset_mutator in matrix["dataset_mutators"].items():
                    self.configurations.append(
                        Config(
                            dataset_name=dataset[0],
                            dataset=dataset[1],
                            method_name=method[0],
                            method=method[1],
                            dataset_mutator_name=dataset_mutator[0],
                            dataset_mutator=dataset_mutator[1],
                            meta=matrix["meta"],
                        )
                    )

    def __iter__(self, *args, **kwargs):
        return self.configurations.__iter__(*args, **kwargs)

    def __len__(self):
        return len(self.configurations)


def run(
    matrix=DEFAULT_MATRIX,
    force_cache=False,
    force_run=False,
    backend="loky",
    abort=True,
    workers=None,
):
    configurations = Configurations(matrix)

    if workers is None:
        workers = os.cpu_count()
        if "sched_getaffinity" in dir(os):
            workers = len(os.sched_getaffinity(0))

    results = ProgressParallel(
        n_jobs=workers // configurations.meta["n_runs"],
        total=len(configurations),
        desc=f"Experiment",
        leave=False,
        backend=backend,
    )(
        delayed(__run_inner)(
            config, force_cache=force_cache, force_run=force_run, abort=abort
        )
        for config in configurations
    )

    return results


def plot(results, plot_robustness=False):
    key = lambda config_result: (
        config_result[0].dataset_name,
        config_result[0].dataset_mutator_name,
    )
    results = sorted(results, key=key)
    groups = groupby(results, key)
    for k, group in groups:
        fig, axes = plt.subplots(1, 4 if plot_robustness else 3, figsize=(18, 4))

        for config, result in group:
            for i, ax in enumerate(axes.flatten()):
                if len(result["x"] > 100):
                    ax.plot(
                        result["x"],
                        result.iloc[:, 1 + i],
                        "-",
                        label=f"{config.method_name}" if i == 0 else "",
                    )
                    ax.fill_between(
                        result["x"],
                        result.iloc[:, 1 + i] - result.iloc[:, 5 + i],
                        result.iloc[:, 1 + i] + result.iloc[:, 5 + i],
                        color="grey",
                        alpha=0.2,
                    )
                else:
                    ax.errorbar(
                        result["x"],
                        result.iloc[:, 1 + i],
                        yerr=result.iloc[:, 5 + i],
                        label=f"{config.method_name}" if i == 0 else "",
                    )
                ax.set_xlabel("Instances")
                ax.set_ylabel(["Accuracy", "F1", "AUC ROC", "Empirical Robustness"][i])
                plt.suptitle(f"{config.dataset_name}")

        fig.legend()


def table(results, tablefmt="fancy_grid"):
    key = lambda config_result: (
        config_result[0].dataset_name,
        config_result[0].dataset_mutator_name,
    )
    results = sorted(results, key=key)
    groups = groupby(results, key)

    def func(result):
        try:
            return int(
                result[result["accuracy_score"].gt(result["accuracy_score"].iloc[-1])]
                .iloc[0]
                .x
            )
        except IndexError:
            return "-"

    for k, group in groups:
        has_err = not math.isnan(results[0][1].iloc[:, 5][0])
        print(k[0])
        print(
            tabulate(
                sorted(
                    [
                        [
                            conf.method_name,
                            f"{skmetrics.auc(result['x'], result.iloc[:,1]):.2f}"
                            + (
                                f"±{skmetrics.auc(result['x'], result.iloc[:,1]+result.iloc[:,5])}"
                                if has_err
                                else ""
                            ),
                            f"{skmetrics.auc(result['x'], result.iloc[:,2]):.2f}"
                            + (
                                f"±{skmetrics.auc(result['x'], result.iloc[:,2]+result.iloc[:,6])}"
                                if has_err
                                else ""
                            ),
                            f"{skmetrics.auc(result['x'], result.iloc[:,3]):.2f}"
                            + (
                                f"±{skmetrics.auc(result['x'], result.iloc[:,3]+result.iloc[:,7])}"
                                if has_err
                                else ""
                            ),
                            func(result),
                        ]
                        for conf, result in group
                    ],
                    key=lambda x: -float(x[1].split("±")[0]),
                ),
                tablefmt=tablefmt,
                headers=["method", "AUC LAC", "AUC LF1C", "AUC AUC ROC", "max % @"],
            )
        )


def __run_inner(config, force_cache=False, force_run=False, backend="loky", abort=None):
    try:
        cached_config, metrics = __read_result(f"cache/{config.serialize()}.csv")
        if force_run:
            raise FileNotFoundError()
        return (cached_config, metrics)

    except FileNotFoundError:
        if force_cache:
            raise Exception(f"Cache file 'cache/{config.serialize()}.csv' not found")

        try:
            metrics = ProgressParallel(
                n_jobs=config.meta["n_runs"],
                total=config.meta["n_runs"],
                desc=f"Run",
                leave=False,
                backend=backend,
            )(
                delayed(
                    lambda dataset, method: MyActiveLearner(
                        metrics=[
                            accuracy_score,
                            f1_score,
                            roc_auc_score,
                            empirical_robustness,
                        ]
                    ).active_learn2(
                        *active_split(
                            *dataset, labeled_size=config.meta["labelled_size"]
                        ),
                        method,
                    )
                )(config.dataset(), config.method)
                for _ in range(config.meta["n_runs"])
            )
        except Exception as e:
            if abort:
                raise e
            print("WARN: Experiment failed, continuing anyway")
            return (config, None)
        metrics = metrics[0].average2(metrics[1:])
        __write_result(config, metrics)

    return (config, metrics)


def __write_result(config, result):
    file = f"cache/{config.serialize()}.csv"
    with open(file, "w") as f:
        json.dump(config.json(), f)
        f.write("\n")
        result.to_csv(f)


def __read_result(file):
    with open(file, "r") as f:
        config = Config(**json.loads(f.readline()))
        result = pd.read_csv(f, index_col=0)
    return (config, result)


def __flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in __flatten_dict(value).items():
                    yield (key, subkey), subvalue
            else:
                yield key, value

    return dict(items())


def __plot_metrics(axes, metrics, stderr, legend):
    for i, ax in enumerate(axes.flatten()):
        ax.errorbar(
            metrics["x"],
            metrics.iloc[:, 1 + i],
            yerr=stderr.iloc[:, 1 + i],
            label=f"{legend}" if i == 0 else "",
        )
        ax.set_xlabel("Instances")
        ax.set_ylabel(["Accuracy", "F1", "AUC ROC", "Empirical Robustness"][i])
        plt.suptitle(dataset_name)


def __progress_hack():
    # Annoying hack so that the progressbars disapear as they're supposed to
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
