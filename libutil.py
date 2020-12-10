import pandas as pd
import numpy as np
from joblib import Parallel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.notebook import tqdm
from art.metrics import empirical_robustness

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
        with tqdm(
            disable=not self._use_tqdm,
            total=self._total,
            leave=self.leave,
            desc=self.desc,
        ) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
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
        self.frame = pd.DataFrame({"x": [], **{f.__name__: [] for f in metrics}})

    def collect(self, x, clf, labels, test_set):
        """
        Collect metrics from the classifier using a particular test set and marking the point at x.
        """

        result = {}
        for metric in self.metrics:
            if metric == f1_score:
                result[metric.__name__] = metric(
                    labels, clf.predict(test_set), average='micro' if len(np.unique(labels)) > 2 else 'binary'
                )
            elif metric == roc_auc_score:
                if len(np.unique(labels)) > 2:
                    result[metric.__name__] = roc_auc_score(
                        labels, clf.predict_proba(test_set), multi_class='ovr'
                    )
                else:
                    result[metric.__name__] = roc_auc_score(labels, clf.predict_proba(test_set)[:,1])
            elif metric == empirical_robustness:
                result[metric.__name__] = empirical_robustness(clf, x, 'fgm', attack_params={eps=0.2}) 
            else:
                result[metric.__name__] = metric(labels, clf.predict(test_set))
                    
        self.frame = self.frame.append({"x": x, **result}, ignore_index=True)

    def average(self, others):
        merged = pd.concat([self.frame] + [other.frame for other in others])
        averaged = merged.groupby(merged.index).mean()
        sem = merged.groupby(merged.index).sem()
        return averaged, sem

def generate_query_synthesis():
    """
    Generate data and an oracle for query synthesis active learning.
    
    Returns:
    * An initial training set X
    * Labels for X
    * An oracle 
    """