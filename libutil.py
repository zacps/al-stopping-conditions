import os
import socket
import typing
import requests

from contextlib import contextmanager
from functools import wraps

import scipy
import pandas as pd
import numpy as np
from joblib import Parallel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.notebook import tqdm
from art.metrics import empirical_robustness
from art.estimators.classification.scikitlearn import ScikitlearnSVC


COLORS = ["#56ad94", "#8f56cf", "#365300", "#a96aa5", "#bf6148"]


class ProgressParallel(Parallel):
    def __init__(
        self, use_tqdm=True, total=None, desc=None, leave=True, *args, **kwargs
    ):
        self._use_tqdm = use_tqdm
        self._total = total
        self.desc = desc
        self.leave = leave
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if run_from_ipython():
            with tqdm(
                disable=not self._use_tqdm,
                total=self._total,
                leave=self.leave,
                desc=self.desc,
            ) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)
        else:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if run_from_ipython():
            if self._total is None:
                self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()


class Metrics:
    def __init__(self, name=None, metrics=None):
        if metrics is None:
            metrics = [accuracy_score, f1_score, roc_auc_score]
        self.name = name
        self.metrics = metrics
        self.frame = pd.DataFrame(
            {
                "x": [],
                **{f.__name__: [] for f in metrics if not isinstance(f, str)},
                **{name: [] for name in metrics if isinstance(name, str)},
            }
        )

    def collect(
        self, x, clf, labels, test_set, X_unlabelled=None, unique_labels=None, **kwargs
    ):
        """
        Collect metrics from the classifier using a particular test set and marking the point at x.
        """

        result = {}
        unique_labels = (
            unique_labels if unique_labels is not None else np.unique(labels)
        )

        predict = clf.predict(test_set)
        predict_proba = clf.predict_proba(test_set)

        for metric in self.metrics:
            if metric == f1_score:
                result[metric.__name__] = f1_score(
                    labels,
                    predict,
                    average="micro" if len(unique_labels) > 2 else "binary",
                    pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1,
                )
            elif metric == roc_auc_score:
                if len(np.unique(labels)) > 2 or len(labels.shape) > 1:
                    result[metric.__name__] = roc_auc_score(
                        labels, predict_proba, multi_class="ovr"
                    )
                else:
                    result[metric.__name__] = roc_auc_score(labels, predict_proba[:, 1])
            elif metric == empirical_robustness:
                result[metric.__name__] = empirical_robustness(
                    ScikitlearnSVC(clf), test_set, "fgsm", attack_params={"eps": 0.2}
                )
            elif metric == "accuracy_train":
                result[metric.__name__] = accuracy_score(labels, predict)
            elif metric == "n_support":
                result[metric] = np.sum(clf.n_support_)
            elif isinstance(metric, str):
                result[metric] = kwargs.get(metric, None)
            else:
                result[metric.__name__] = metric(labels, predict)

        self.frame = self.frame.append({"x": x, **result}, ignore_index=True)

    def average(self, others):
        merged = pd.concat([self.frame] + [other.frame for other in others])
        averaged = merged.groupby(merged.index).mean()
        sem = merged.groupby(merged.index).sem()
        sem.columns = [str(col) + "_stderr" for col in sem.columns]
        return averaged, sem

    def average2(self, others):
        merged = pd.concat([self.frame] + [other.frame for other in others])
        averaged = merged.groupby(merged.index).mean()
        sem = merged.groupby(merged.index).sem()
        sem.columns = [str(col) + "_stderr" for col in sem.columns]
        return pd.concat([averaged, sem], axis=1)


def average(frame, others):
    merged = pd.concat([frame] + [other for other in others])
    averaged = merged.groupby(merged.index).mean()
    sem = merged.groupby(merged.index).sem()
    sem.columns = [str(col) + "_stderr" for col in sem.columns]
    return pd.concat([averaged, sem], axis=1)


def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def out_dir():
    return os.environ["OUT_DIR"]

def webhook():
    return os.environ["WEBHOOK"]

def dataset_dir():
    return os.environ["DATASET_DIR"]


def get_mutator(name: str):
    from nesi_bias import bias

    if name.startswith("noise"):
        from nesi_noise import noise

        amount = int(name[5:]) * 1e-2
        return partial(noise, amount=amount)
    if name.startswith("unbalanced2"):
        from nesi_unbalanced import unbalanced2

        amount = int(name.split("-")[1]) * 1e-2
        return partial(unbalanced2, amount=amount)
    if name.startswith("bias"):
        raise Exception("bias mutator getting not implemented")
    raise Exception(f"unknown mutator {name}")


# Source:
# https://stackoverflow.com/questions/12377013/is-there-a-library-function-in-python-to-turn-a-generator-function-into-a-functi
def listify(fn=None, wrapper=list):
    """
    A decorator which wraps a function's return value in ``list(...)``.

    Useful when an algorithm can be expressed more cleanly as a generator but
    the function should return an list.

    Example::

        >>> @listify
        ... def get_lengths(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths(["spam", "eggs"])
        [4, 4]
        >>>
        >>> @listify(wrapper=tuple)
        ... def get_lengths_tuple(iterable):
        ...     for i in iterable:
        ...         yield len(i)
        >>> get_lengths_tuple(["foo", "bar"])
        (3, 3)
    """

    def listify_return(fn):
        @wraps(fn)
        def listify_helper(*args, **kw):
            return wrapper(fn(*args, **kw))

        return listify_helper

    if fn is None:
        return listify_return
    return listify_return(fn)


def n_cpus():
    cpus = os.cpu_count()
    if "sched_getaffinity" in dir(os):
        cpus = len(os.sched_getaffinity(0))
    return cpus


@contextmanager
def atomic_write(filepath: str, mode: str) -> typing.IO:
    """Writeable file object that atomically updates a file (using a temporary file).

    :param filepath: the file path to be opened
    :param binary: whether to open the file in a binary mode instead of textual
    """

    tmppath = filepath + ".X"
    try:
        with open(tmppath, mode) as file:
            yield file
        os.rename(tmppath, filepath)
    finally:
        try:
            os.remove(tmppath)
        except (IOError, OSError):
            pass


class Notifier:
    def __init__(self, webhook):
        self.webhook = webhook

    def notify(self, data):
        for _i in range(3):
            try:
                requests.post(
                    self.webhook,
                    data={"content": data},
                )
                break
            except ConnectionError as con_err:
                print(f"Couldn't send failed notification because: {con_err}")

    def completion(self, configurations, duration):
        return self.notify(
            f"Run with {len(configurations)} experiments on {socket.gethostname()} completed after {duration/60/60:.1f}h"
        )

    def error(self, configurations, duration, e):
        jobid = os.environ.get("SLURM_JOB_ID", None)
        jobid_str = f" ({jobid})" if jobid is not None else ""
        return self.notify(
            f"Run with {len(configurations)} experiments on {socket.gethostname()} **FAILED** after {duration/60/60:.1f}h{jobid_str}\n```{e}```"
        )
