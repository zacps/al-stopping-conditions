import os
import math
import pickle
import socket
from time import monotonic
from dataclasses import dataclass
from typing import Dict, Any, Callable
import json
import math
from itertools import groupby
import glob
from pprint import pprint, pformat

import requests
import pandas as pd
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
from tabulate import tabulate

from libactive import active_split, MyActiveLearner, CompressedStore


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
    
    def __repr__(self):
        return pformat(self.json())


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
                        
    def __repr__(self):
        return pformat({"meta": self.meta.__repr__(), "configurations": self.configurations.__repr__()})

    def __iter__(self, *args, **kwargs):
        return self.configurations.__iter__(*args, **kwargs)
    
    def __getitem__(self, *args, **kwargs):
        return self.configurations.__getitem__(*args, **kwargs)

    def __len__(self):
        return len(self.configurations)


# TODO: Add profiling support?
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
    dry_run=False
):
    if fragment_id is None:
        __progress_hack()
    configurations = Configurations(matrix)
    start = monotonic()
    
    # For NeSI
    if fragment_id is not None:
        configurations.configurations = configurations.configurations[fragment_id:fragment_id+fragment_length]
        
    # Monomorphise parametric meta parameters
    for config in configurations:
        for k, v in config.meta.items():
            if isinstance(v, dict):
                config.meta[k] = v.get(config.dataset_name, v["*"])
                
    # Exit if this is a dry run
    if dry_run:
        print("Exiting due to dry run:")
        pprint(configurations)
        print(f"Runs: {fragment_run_start}-{fragment_run_end}")
        return

    # Detect number of available CPUs
    if workers is None:
        workers = os.cpu_count()
        if "sched_getaffinity" in dir(os):
            workers = len(os.sched_getaffinity(0))

    if fragment_run_start is not None:
        n_runs = (fragment_run_end-fragment_run_start) if fragment_run_end is not None else 1
    else:
        n_runs = configurations.meta['n_runs']

    try:
        results = ProgressParallel(
            n_jobs=math.ceil(workers / n_runs),
            total=len(configurations),
            desc=f"Experiment",
            leave=False,
            backend=backend,
        )(
            delayed(__run_inner)(
                config, force_cache=force_cache, force_run=force_run, abort=abort, metrics_measures=metrics, workers=workers, fragment_run_start=fragment_run_start, fragment_run_end=fragment_run_end
            )
            for config in configurations
        )
        if configurations.meta['ret_classifiers']:
            for i, config in enumerate(configurations):
                results[i] = (results[i], [__read_classifiers(config, j) for j in range(n_runs)])
    except Exception as e:
        duration = monotonic()-start
        
        requests.post(
            'https://discord.com/api/webhooks/809248326485934080/aIHL726wKxk42YpDI_GtjsqfAWuFplO3QrXoza1r55XRT9-Ao9Rt8sBtexZ-WXSPCtsv', 
            data={'content': f"Run with {len(configurations)} experiments on {socket.gethostname()} **FAILED** after {duration/60/60:.1f}h\n```{e}```"}
        )
        raise e
    
    duration = monotonic()-start
    
    if duration > 10*60 or fragment_id is not None:
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
            if isinstance(result, list):
                result = result[0].average2(result[1:])
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


def __run_inner(config, force_cache=False, force_run=False, backend="loky", abort=None, metrics_measures=None, workers=None, fragment_run_start=None, fragment_run_end=None):
    if metrics_measures is None:
        metrics_measures = [
            accuracy_score,
            f1_score,
            roc_auc_score,
            #empirical_robustness,
            "time"
        ]
    
    try:
        try:
            cached_config, metrics = __read_result(f"{out_dir()}/{config.serialize()}.csv", config)
        except FileNotFoundError as e:
            if config.model_name == None or config.model_name == "svm-linear":
                cached_config, metrics = __read_result(f"{out_dir()}/{config.serialize_no_model()}.csv", config)
                cached_config.model_name = "svm-linear"
            else:
                raise e
        if force_run:
            raise FileNotFoundError()
        return (cached_config, metrics)

    except (FileNotFoundError, EOFError, pd.errors.EmptyDataError):
        if force_cache:
            raise Exception(f"Cache file '{out_dir()}/{config.serialize()}.csv' not found")
            
        if workers is None:
            workers = os.cpu_count()
            if "sched_getaffinity" in dir(os):
                workers = len(os.sched_getaffinity(0))

        try:
            # Seed a random state generator. This seed is constant between methods/datasets/models so comparisons can be made with fewer runs.
            # It is however *variant* with each run.
            if fragment_run_start is not None:
                if fragment_run_end is not None:
                    runs = list(range(fragment_run_start, fragment_run_end+1))
                else:
                    runs = [fragment_run_start]
            else:
                runs = range(config.meta["n_runs"])
            random_state = [check_random_state(i) for i in runs]
            
            metrics = ProgressParallel(
                n_jobs=min(config.meta["n_runs"], workers),
                total=config.meta["n_runs"],
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
                            i=i
                        ),
                        method,
                        metrics=metrics_measures,
                        model=config.model_name.lower(),
                        ret_classifiers=config.meta.get("ret_classifiers", False),
                        stop_info=config.meta.get("stop_info", False),
                        stop_function=config.meta.get("stop_function", ("default", lambda learner: False))[1],
                        config_str=config.serialize(),
                        i=i,
                        pool_subsample=config.meta.get("pool_subsample", None),
                        ee=config.meta.get("ee", "offline")
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
        __write_result(config, metrics)
        for i in range(config.meta['n_runs']):
            try:
                os.remove(f"{out_dir()}/runs/{config.serialize()}_{i}.csv")
            except FileNotFoundError:
                pass

    
    return (config, metrics)


def __write_result(config, result):
    if isinstance(result, list):
        for i in range(len(result)):
            file = f"{out_dir()}/{config.serialize()}_{i}.csv"
            with open(file, "w") as f:
                json.dump(config.json(), f)
                f.write("\n")
                result[i].frame.to_csv(f)
    else:
        file = f"{out_dir()}/{config.serialize()}.csv"
        with open(file, "w") as f:
            json.dump(config.json(), f)
            f.write("\n")
            result.to_csv(f)
        
        
def __read_classifiers(config, i=None):
    pfile = f"{out_dir()}/classifiers/{config.serialize()}.pickle"
    zfile = f"{out_dir()}/classifiers/{config.serialize()}_{i}.zip"
    try:
        with open(pfile, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return CompressedStore(zfile, read=True)

def __read_result(file, config):
    if config.meta.get("aggregate", True):
        with open(file, "r") as f:
            cached_config = Config(**{"model_name": "svm-linear", **json.loads(f.readline())})
            result = pd.read_csv(f, index_col=0)
        return (cached_config, result)
    else:
        results = []
        for name in glob.glob(f"{out_dir()}/{config.serialize()}_*.csv"):
            with open(name, "r") as f:
                cached_config = Config(**{"model_name": "svm-linear", **json.loads(f.readline())})
                results.append(pd.read_csv(f, index_col=0))
        if len(results) != config.meta.get("n_runs", 10):
            raise FileNotFoundError("did not find disaggregated runs")
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
