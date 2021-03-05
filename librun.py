import os
import io
import itertools
import math
import pickle
import socket
import inspect
from time import monotonic
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Callable
from functools import partial
import json
import math
from itertools import groupby
import glob

import requests
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import HTML, display
from libutil import ProgressParallel
from joblib import delayed
from modAL import batch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn import metrics as skmetrics
from sklearn.utils import check_random_state
from art.metrics import empirical_robustness
from tabulate import tabulate

from libactive import active_split, MyActiveLearner, CompressedStore
from libadversarial import random_batch


@dataclass
class Config:
    dataset_name: str
    method_name: str
    dataset_mutator_name: str
    model_name: str
    meta: Dict["str", Any]
    method: Callable = None
    dataset: Callable = None
    dataset_mutator: Callable = None

    def serialize(self):
        meta_str = "__".join([f"{k}={v}" if k != "stop_function" else f"{k}={v[0]}" for k, v in self.meta.items()])
        return f"{self.dataset_name}__{self.dataset_mutator_name}__{self.method_name}__{self.model_name}__{meta_str}"
    
    def serialize_no_model(self):
        meta_str = "__".join([f"{k}={v}" if k != "stop_function" else f"{k}={v[0]}" for k, v in self.meta.items()])
        return f"{self.dataset_name}__{self.dataset_mutator_name}__{self.method_name}__{meta_str}"

    def json(self):
        return {
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "dataset_mutator_name": self.dataset_mutator_name,
            "model_name": self.model_name,
            "meta": {k: v if k != "stop_function" else v[0] for k, v in self.meta.items()},
        }


class Configurations:
    def __init__(self, matrix):
        self.configurations = []
        self.meta = matrix["meta"]

        for dataset in matrix["datasets"]:
            for method in matrix["methods"]:
                for model in matrix["models"]:
                    for dataset_mutator in matrix["dataset_mutators"].items():
                        self.configurations.append(
                            Config(
                                dataset_name=dataset[0],
                                dataset=dataset[1],
                                method_name=method[0],
                                method=method[1],
                                dataset_mutator_name=dataset_mutator[0],
                                dataset_mutator=dataset_mutator[1],
                                model_name=model,
                                meta=matrix["meta"],
                            )
                        )

    def __iter__(self, *args, **kwargs):
        return self.configurations.__iter__(*args, **kwargs)

    def __len__(self):
        return len(self.configurations)


def run(
    matrix,
    force_cache=False,
    force_run=False,
    backend="loky",
    abort=True,
    workers=None,
    metrics=None,
    fragment_id=None,
    fragment_length=1
):
    if fragment_id is None:
        __progress_hack()
    configurations = Configurations(matrix)
    start = monotonic()
    
    # For NeSI
    if fragment_id is not None:
        configurations = configurations[fragment_id:fragment_id+fragment_length]

    if workers is None:
        workers = os.cpu_count()
        if "sched_getaffinity" in dir(os):
            workers = len(os.sched_getaffinity(0))

    try:
        results = ProgressParallel(
            n_jobs=math.ceil(workers / configurations.meta["n_runs"]),
            total=len(configurations),
            desc=f"Experiment",
            leave=False,
            backend=backend,
        )(
            delayed(__run_inner)(
                config, force_cache=force_cache, force_run=force_run, abort=abort, metrics_measures=metrics, workers=workers
            )
            for config in configurations
        )
        if configurations.meta['ret_classifiers']:
            classifiers = [__read_classifiers(config, i) for i, config in enumerate(configurations)]
            results = list(zip(results, classifiers))
    except Exception as e:
        duration = monotonic()-start
        
        top = inspect.stack()[1]
        filename = os.path.basename(top.filename)
        requests.post(
            'https://discord.com/api/webhooks/809248326485934080/aIHL726wKxk42YpDI_GtjsqfAWuFplO3QrXoza1r55XRT9-Ao9Rt8sBtexZ-WXSPCtsv', 
            data={'content': f"Run with {len(configurations)} experiments on {socket.gethostname()} **FAILED** after {duration/60/60:.1f}h\n```{e}```"}
        )
        raise e
    
    duration = monotonic()-start
    
    if duration > 10*60 or fragment_id is not None:
        top = inspect.stack()[1]
        filename = os.path.basename(top.filename)
        requests.post(
            'https://discord.com/api/webhooks/809248326485934080/aIHL726wKxk42YpDI_GtjsqfAWuFplO3QrXoza1r55XRT9-Ao9Rt8sBtexZ-WXSPCtsv', 
            data={'content': f"Run with {len(configurations)} experiments on {socket.gethostname()} completed after {duration/60/60:.1f}h"}
        )

    return results


def plot(results, plot_robustness=False, key=None, series=None, title=None, ret=False, sort=True, figsize=(18,4), extra=0):
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
        fig, axes = plt.subplots(1, (4 if plot_robustness else 3)+extra, figsize=figsize)
        figaxes.append((fig, axes))

        for config, result in group:
            for i, ax in enumerate(axes.flatten()[:-extra if extra != 0 else len(axes.flatten())]):
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
                ax.set_xlabel("Instances")
                ax.set_ylabel(["Accuracy", "F1", "AUC ROC", "Empirical Robustness"][i])
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
            upper = result[metric]+result[f"{metric}_stderr"]
            norm = result[result[metric].ge(result[metric].iloc[-1])].iloc[0].x
            return f"{norm:.0f}" + f"±{abs(result[upper.ge(result[metric].iloc[-1])].iloc[0].x-norm):.0f}" if has_err else ""
        except IndexError:
            return "-"
        
    def area_under(result, has_err, metric, baseline=1):
        return f"{skmetrics.auc(result['x'], result[metric])/baseline:.4f}" + (
            f"±{(skmetrics.auc(result['x'], result[metric]+result[metric+'_stderr'])-skmetrics.auc(result['x'], result[metric]))/baseline:.0g}"
            if has_err
            else ""
        )
    def baseline(group, metric):
        return next((skmetrics.auc(result['x'], result[metric]) for conf, result in group if conf.method_name == "random"), 1)

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
                            area_under(result, has_err, "accuracy_score", baseline=baseline(group, "accuracy_score")),
                            area_under(result, has_err, "f1_score", baseline=baseline(group, "f1_score")),
                            area_under(result, has_err, "roc_auc_score", baseline=baseline(group, "roc_auc_score")),
                            max_at(result, has_err, "accuracy_score"),
                            max_at(result, has_err, "roc_auc_score"),
                            result.time.sum() if 'time' in result else None
                        ]
                        for conf, result in group
                    ],
                    key=lambda x: -float(x[1].split("±")[0]),
                ),
                tablefmt=tablefmt,
                headers=["method", "AUC LAC", "AUC LF1C", "AUC AUC ROC", "Instances to max accuracy", "Instances to max AUC ROC", "Time"],
            )
        )


def __run_inner(config, force_cache=False, force_run=False, backend="loky", abort=None, metrics_measures=None, workers=None):
    if metrics_measures is None:
        metrics_measures = [
            accuracy_score,
            f1_score,
            roc_auc_score,
            #empirical_robustness,
            "time"
        ]
        
    # Monomorphise parametric meta parameters
    for k, v in config.meta.items():
        if isinstance(v, dict):
            config.meta[k] = v.get(config.dataset_name, v["*"])
    
    try:
        try:
            cached_config, metrics = __read_result(f"cache/{config.serialize()}.csv", config)
        except FileNotFoundError as e:
            if config.model_name == None or config.model_name == "svm-linear":
                cached_config, metrics = __read_result(f"cache/{config.serialize_no_model()}.csv", config)
                cached_config.model_name = "svm-linear"
            else:
                raise e
        if force_run:
            raise FileNotFoundError()
        return (cached_config, metrics)

    except (FileNotFoundError, EOFError, pd.errors.EmptyDataError):
        if force_cache:
            raise Exception(f"Cache file 'cache/{config.serialize()}.csv' not found")
            
        if workers is None:
            workers = os.cpu_count()
            if "sched_getaffinity" in dir(os):
                workers = len(os.sched_getaffinity(0))

        try:
            # Seed a random state generator. This seed is constant between methods/datasets/models so comparisons can be made with fewer runs.
            # It is however *variant* with each run.
            random_state = check_random_state(42)
            
            metrics = ProgressParallel(
                n_jobs=min(config.meta["n_runs"], workers),
                total=config.meta["n_runs"],
                desc=f"Run",
                leave=False,
                backend=backend,
            )(
                delayed(
                    lambda dataset, method, i: MyActiveLearner(
                        metrics=metrics_measures
                    ).active_learn2(
                        # It's important that the split is re-randomised per run.
                        *active_split(
                            *dataset, 
                            labeled_size=config.meta.get("labelled_size", 0.1), 
                            test_size=config.meta.get("test_size", 0.5), 
                            ensure_y=config.meta.get("ensure_y", False), 
                            random_state=random_state,
                            mutator=config.dataset_mutator,
                            config_str=config.serialize(),
                            i=i
                        ),
                        method,
                        model=config.model_name.lower(),
                        ret_classifiers=config.meta.get("ret_classifiers", False),
                        stop_info=config.meta.get("stop_info", False),
                        stop_function=config.meta.get("stop_function", ("default", lambda learner: False))[1],
                        config_str=config.serialize(),
                        i=i,
                        pool_subsamble=config.meta.get("pool_subsample", None)
                    )
                )(config.dataset(), config.method, i)
                for i in range(config.meta["n_runs"])
            )
            
        except Exception as e:
            if abort:
                raise e
            print("WARN: Experiment failed, continuing anyway")
            return (config, None)
        
        if config.meta.get("aggregate", True):
            metrics = metrics[0].average2(metrics[1:])
        __write_result(config, metrics)
        for i in range(config.meta['n_runs']):
            try:
                os.remove(f"cache/runs/{config.serialize()}_{i}.csv")
            except FileNotFoundError:
                pass

    
    return (config, metrics)


def __write_result(config, result):
    if isinstance(result, list):
        for i in range(len(result)):
            file = f"cache/{config.serialize()}_{i}.csv"
            with open(file, "w") as f:
                json.dump(config.json(), f)
                f.write("\n")
                result[i].frame.to_csv(f)
    else:
        file = f"cache/{config.serialize()}.csv"
        with open(file, "w") as f:
            json.dump(config.json(), f)
            f.write("\n")
            result.to_csv(f)
        
        
def __write_classifiers(config, classifiers):
    file = f"cache/classifiers/{config.serialize()}.pickle"
    with open(file, "wb") as f:
        pickle.dump(classifiers, f)
        
def __read_classifiers(config, i=None):
    pfile = f"cache/classifiers/{config.serialize()}.pickle"
    zfile = f"cache/classifiers/{config.serialize()}_{i}.zip"
    try:
        with open(pfile, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return CompressedStore(zfile)

def __read_result(file, config):
    if config.meta.get("aggregate", True):
        with open(file, "r") as f:
            config = Config(**{"model_name": "svm-linear", **json.loads(f.readline())})
            result = pd.read_csv(f, index_col=0)
        return (config, result)
    else:
        results = []
        for name in glob.glob(f"cache/{config.serialize()}_*.csv"):
            with open(name, "r") as f:
                config = Config(**{"model_name": "svm-linear", **json.loads(f.readline())})
                results.append(pd.read_csv(f, index_col=0))
        return config, results


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


def __free_cpus(pesimistic=False):
    """
    Count the number of free CPU cores.
    """
    rounder = math.floor if pesimistic else math.ceil
    return rounder((100*psutil.cpu_count()-psutil.cpu_percent(interval=1))/100)