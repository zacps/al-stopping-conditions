"""
Defines active learning stopping criteria.

The standard signature for a criteria is:

```python
def stop(**metrics, **parameters) -> bool:
    pass
```

Where `metrics` is a `dict` with the following keys:

* `uncertainty_average` Average of classifier uncertainty on unlabelled pool
* `uncertainty_min` Minimum of classifier uncertainty on unlabelled pool
* `n_support` Number of support vectors
* `expected_error` Expected error on unlabelled pool
* `last_accuracy` Accuracy of the last classifier on the current query set (oracle accuracy)
* `contradictory_information`
* `stop_set` Predictions on the stop set

All values are lists containing all evaluations up to the current classifier. The most recent is in the last element of the list.

---

Fundamentally all stop conditions take a metric and try to determine if the current value is:

* A minimum
* A maximum
* Constant
* At a threshold

The first three require approximation, this is implemented by:

* `__is_approx_constant`
* `__is_approx_minimum`

Which take a threshold and a number of iterations for which the value should be stable.

"""

from collections import namedtuple
import time
import datetime
import itertools
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
import operator
from typing import Callable, Dict, List, Type

import dill
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from scipy.sparse import data
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances, pairwise_kernels
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
from libactive import active_split, delete_from_csr
from tabulate import tabulate
from statsmodels.stats.inter_rater import fleiss_kappa
from autorank import autorank, plot_stats
from tvregdiff.tvregdiff import TVRegDiff
from modAL.uncertainty import classifier_entropy

import libdatasets
from libutil import out_dir, listify
from libconfig import Config


logger = logging.getLogger(__name__)


class FailedToTerminate(Exception):
    def __init__(self, method):
        super().__init__(f"{method} failed to determine a stopping location")


class InvalidAssumption(Exception):
    def __init__(self, method, reason):
        super().__init__(f"{method} could not be evaluated because: {reason}")


class Criteria(ABC):
    @abstractmethod
    def metric(self, **kwargs):
        pass

    @abstractmethod
    def condition(self, x, metric):
        pass

    @property
    def display_name(self) -> str:
        return getattr(self, "display_name", type(self).__name__)

    @classmethod
    def all_criteria(cls) -> List[Type]:
        # https://stackoverflow.com/questions/3862310/how-to-find-all-the-subclasses-of-a-class-given-its-name
        return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in cls.all_criteria(c)])


@dataclass
class SC_entropy_mcs(Criteria):
    """
    Determine a stopping point based on when *all* samples in the unlabelled pool are
    below a threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    threshold: int = 0.01

    def metric(self, entropy_max, **kwargs):
        return entropy_max

    def condition(self, x, metric):
        if not (metric <= self.threshold).any():
            raise FailedToTerminate("SC_entropy_mcs")
        return x.iloc[np.argmax(metric <= self.threshold)]


@dataclass
class SC_oracle_acc_mcs(Criteria):
    """
    Determine a stopping point based on when the accuracy of the current classifier on the
    just-labelled samples is above a threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    threshold: int = 0.9

    def metric(self, classifiers, **kwargs):
        return acc_last(classifiers)[1]


    def condition(self, x, metric):
        """
        Determine a stopping point based on when the accuracy of the current classifier on the
        just-labelled samples is above a threshold.

        https://www.aclweb.org/anthology/I08-1048.pdf
        """

        try:
            return x[1:][np.argmax(np.array(metric)[1:] >= self.threshold) + 1]
        except IndexError:
            raise FailedToTerminate("SC_oracle_acc_mcs")


@dataclass
class SC_mes(Criteria):
    """
    Determine a stopping point based on the expected error of the classifier is below a
    threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    threshold: int = 1e-2

    def metric(self, expected_error_min, **kwargs):
        return expected_error_min

    def condition(self, x, metric):
        if (metric <= self.threshold).any():
            return x.iloc[np.argmax(metric <= self.threshold)]

        raise FailedToTerminate("SC_mes")


class EVM(Criteria):
    """
    Determine a stopping point based on the variance of the uncertainty in the unlabelled pool.

    * `selected` determines if the variance is calculated accross the entire pool or only the instances
    to be selected.

    * `n` the number of values for which the variance must decrease sequentially, paper used `2`.
    * `m` the threshold for which the variance must decrease by to be considered decreasing, paper
    used `0.5`.

    https://www.aclweb.org/anthology/W10-0101.pdf
    """

    selected: bool = True
    n: int = 2
    m: float = 1e-3


    def metric(self, uncertainty_variance, uncertainty_variance_selected, **kwargs):
        return uncertainty_variance_selected if self.selected else uncertainty_variance

    def condition(self, x, metric):
        current = 0
        last = metric[0]
        for i, value in enumerate(metric[1:]):
            if current == self.n:
                # This used to be -1, which was a bug I think?
                return x[i + 1]
            if value < last - self.m:
                current += 1
            last = value
        raise FailedToTerminate("EVM")


class VM(EVM):
    m: float = 0


@dataclass
class SSNCut(Criteria):
    """
    NOTES:
    * They used an RBF svm to match the RBF affinity measure for SpectralClustering
    * As we carry out experiments on a linear svm we also use a ~~linear~~ *cosine* affinity matrix (by default)
    * Cosine is used as our features are not necessarilly normalized

    file:///F:/Documents/Zotero/storage/DJGRDXSK/Fu%20and%20Yang%20-%202015%20-%20Low%20density%20separation%20as%20a%20stopping%20criterion%20for.pdf
    """

    m: float = 0.2

    def metric(
        self,
        classifiers,
        **kwargs,
    ):

        if any(getattr(clf.estimator, "kernel", None) != "linear" for clf in classifiers):
            raise InvalidAssumption("SSNCut", "model is not a linear SVM")

        unique_y = np.unique(classifiers[0].y_training)
        if len(unique_y) > 2:
            raise InvalidAssumption("SSNCut", "dataset is not binary")

        clustering = SpectralClustering(n_clusters=unique_y.shape[0], affinity='precomputed')

        out = []

        for i, clf in enumerate(classifiers):
            # Note: With non-binary classification the value of the decision function is a transformation of the distance...
            order = np.argsort(np.abs(clf.estimator.decision_function(clf.X_unlabelled)))
            M = clf.X_unlabelled[order[: min(1000, int(self.m * clf.X_unlabelled.shape[0]))]]

            y0 = clf.predict(M)

            # We use cos+1 as our affinity matrix as we want something that:
            # * Is linear, to fit the rbf svm/rbf affinity pattern in the paper
            # * Handles non-normalized samples (rules out linear)
            # * Is non-negative (condition of scikit-learn's implementation)
            affinity = pairwise_kernels(M, metric="cosine")+1
            assert np.all(affinity >= 0)
            y1 = clustering.fit_predict(affinity)

            y0 = LabelEncoder().fit_transform(y0)
            diff = np.sum(y0 == y1) / clf.X_unlabelled.shape[0]
            if diff > 0.5:
                diff = 1 - diff
            out.append(diff)

        return out

    def condition(self, x, metric):
        smallest = np.infty
        x0 = 0
        for i, v in enumerate(metric):
            if v < smallest:
                smallest = v
                x0 = 0
            else:
                x0 += 1
            if x0 == 10:
                return x.iloc[i - 10]  # return to the point of lowest value

        raise FailedToTerminate("SSNCut")


@dataclass
class ContradictoryInformation(Criteria):
    """
    Stop when contradictory information drops for `rounds` consecutive rounds.

    https://www-sciencedirect-com.ezproxy.auckland.ac.nz/science/article/pii/S088523080700068X
    """

    rounds: int = 3

    def metric(self, contradictory_information, **kwargs):
        return contradictory_information

    def condition(self, x, metric):
        current = 0
        last = metric[0]
        for i, value in enumerate(metric[1:]):
            if current == self.rounds:
                return x[i + 1]
            if value < last:
                current += 1
            last = value
        raise FailedToTerminate("contradictory_information")


@dataclass
class PerformanceConvergence(Criteria):
    """
    We use k=10 instead of 100 because batch size is 10 for us, 1 for them
    multiple thresholds tested by authors (1e-2, 5e-5)

    https://www.aclweb.org/anthology/C08-1059.pdf
    """

    k: int = 10
    threshold: float = 5e-5
    average: Callable = np.mean
    weak_threshold: float = 0.8

    @listify
    def metric(self, classifiers, **kwargs):
        for clf in classifiers:
            X_subsampled = clf.X_unlabelled[
                np.random.choice(
                    clf.X_unlabelled.shape[0], min(clf.X_unlabelled.shape[0], 1000), replace=False
                )
            ]

            p = clf.predict_proba(X_subsampled)
            # d indicates if a particular prediction is from the winning (max probability) class
            d = (p.T == np.max(p, axis=1)).T * 1

            def tp(p, d):
                "Sum of the probabilities of the winning predictions"
                return np.sum(p * d)

            def fp(p, d):
                "Sum of (1-prob) for the winning prediction"
                return np.sum((1 - p) * d)

            def fn(p, d):
                return np.sum(p * (1 - d))

            yield 2 * tp(p, d) / (2 * tp(p, d) + fp(p, d) + fn(p, d))
            
    def weak_determine_start(self, metric):
        return np.argmax(np.array(metric) >= self.weak_threshold)

    def condition(self, x, metric):
        windows = np.lib.stride_tricks.sliding_window_view(metric, self.k)
        for i in range(1, len(windows)):
            w2 = self.average(windows[i])
            w1 = self.average(windows[i - 1])
            g = w2 - w1
            if windows[i][-1] > np.max(metric[: i + self.k - 1]) and g > 0 and g < self.threshold:
                return x.iloc[i + self.k-1]

        raise FailedToTerminate("performance_convergence")
        
        
@dataclass
class SecondDiffZeroPerformanceConvergence(PerformanceConvergence):
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    wait_iters: int = 5
    start_threshold: float = 0.1
    start_method: str = "1"
        
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        sec = np.array(no_ahead_tvregdiff(grad, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad, sec
    
    def determine_start(self, sec):
        return np.argmax(sec >= self.start_threshold)
    
    def condition(self, x, metric):
        met, grad, sec = metric
        
        if self.start_method == "1":
            start = self.determine_start(sec)
        else:
            start = self.weak_determine_start(met)
            
        if not (sec[start:] <= 0).any():
            raise FailedToTerminate('SecondDiffZeroPerformanceConvergence')
        
        # stop when we hit zero
        return x.iloc[np.argmax(sec[start:] <= 0)+start]
    
    
@dataclass
class FirstDiffZeroPerformanceConvergence(PerformanceConvergence):
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    wait_iters: int = 5
    start_threshold: float = 1e-1
    start_iters: int = 5
    start_method: str = "1"
        
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad
    
    def determine_start(self, grad):
        # Find nth value where the threshold is exceeded
        if np.count_nonzero(grad >= self.start_threshold) < self.start_iters:
            raise FailedToTerminate('FirstDiffZeroPerformanceConvergence')
        return np.searchsorted(np.cumsum(grad >= self.start_threshold), self.start_iters)
    
    def condition(self, x, metric):
        met, grad = metric
        
        if self.start_method == "1":
            start = self.determine_start(grad)
        else:
            start = self.weak_determine_start(met)
            
        if not (grad[start:] >= 0).any():
            raise FailedToTerminate('FirstDiffZeroPerformanceConvergence')
        
        # stop when we hit zero
        return x.iloc[np.argmax(grad[start:] <= 0)+start]


@dataclass
class UncertaintyConvergence(Criteria):
    """
    Stop based on the gradient of the last selected instance. The last selected instance supposedly has maximum uncertainty
    and is hence the most informative.

    We use k=10 instead of 100 because batch size is 10 for us, 1 for them
    multiple thresholds tested by authors (1e-2, 5e-5)

    Metrics:
    * classifier_entropy
    * classifier_margin
    * classifier_minmax

    Threshold:
    * 0.00005

    https://www.aclweb.org/anthology/C08-1059.pdf
    """

    threshold: float = 5e-5
    k: int = 10
    average: Callable = np.median

    def metric(self, classifiers, **kwargs):
        # Maximum entropy in the selected batch of instances
        return metric_selected(classifiers, classifier_uncertainty, aggregator=np.max)

    def condition(self, x, metric):
        metric = 1 - np.array(metric)
        windows = np.lib.stride_tricks.sliding_window_view(metric, self.k)
        for i in range(1, len(windows)):
            w2 = self.average(windows[i])
            w1 = self.average(windows[i - 1])
            g = w2 - w1
            if windows[i][-1] > np.max(metric[: i + self.k - 1]) and g > 0 and g < self.threshold:
                return x.iloc[i + self.k - 1]

        raise FailedToTerminate("UncertaintyConvergence")


@dataclass
class OverallUncertainty(Criteria):
    """
    Stop if the overall uncertainty on the unlabelled pool is less than a threshold.

    https://www.aclweb.org/anthology/C08-1142.pdf
    """
    
    threshold: float = 1e-2
    weak_threshold: float = 1.5e-1

    def metric(self, uncertainty_average, **kwargs):
        return uncertainty_average
    
    def threshold_determine_start(self, metric):
        "Determine a point safe to start evaluating unstable conditions"
        return np.argmax(metric < self.weak_threshold)

    def condition(self, x, metric):
        if not (metric < self.threshold).any():
            raise FailedToTerminate("overall_uncertainty")
        return x.iloc[np.argmax(metric < self.threshold)]
    
    
@dataclass
class FirstDiffMinOverallUncertainty(OverallUncertainty):
    """
    Modified OverallUncertainty which stops when the first derivative hits an (estimated) global
    minimum.
    
    * eps, diffkernel control the regularlization of the differentiation algorithm
    * stop_iters determines how many rounds we check to see if we have a minimum
    """
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    stop_iters: int = 5
    
    start_method: str = "1"
    
    start_threshold: float = -1e-1
    start_iters: int = 5
    
        
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad
    
    def determine_start(self, grad):
        # Find nth value where the threshold is exceeded
        if np.count_nonzero(grad <= self.start_threshold) < self.start_iters:
            raise FailedToTerminate('FirstDiffMinOverallUncertainty')
        return np.searchsorted(np.cumsum(grad <= self.start_threshold), self.start_iters)
        
    def condition(self, x, metric):
        met, grad = metric
        
        if self.start_method == "1":
            start = self.determine_start(grad)
        else:
            start = self.threshold_determine_start(met)
        
        minimum = 1e-1
        iters = self.stop_iters+1 # Will not terminate until min set
        for i, v in enumerate(grad[start:]):
            if iters == self.stop_iters:
                return x.iloc[i+start]
            if v < minimum:
                minimum = v
                iters = 0
            else:
                iters += 1
        raise FailedToTerminate('FirstDiffMinOverallUncertainty')
        
        
@dataclass
class FirstDiffZeroOverallUncertainty(OverallUncertainty):
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    wait_iters: int = 5
        
    start_method: str = "1"
        
    start_threshold: float = -1e-1
    start_iters: int = 5
        
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad
    
    def determine_start(self, grad):
        # Find nth value where the threshold is exceeded
        if np.count_nonzero(grad <= self.start_threshold) < self.start_iters:
            raise FailedToTerminate('FirstDiffZeroOverallUncertainty')
        return np.searchsorted(np.cumsum(grad <= self.start_threshold), self.start_iters)
    
    def condition(self, x, metric):
        met, grad = metric
        
        if self.start_method == "1":
            start = self.determine_start(grad)
        else:
            start = self.threshold_determine_start(met)
            
        if not (grad[start:] >= 0).any():
            raise FailedToTerminate('FirstDiffZeroOverallUncertainty')
        
        # stop when we hit zero
        return x.iloc[np.argmax(grad[start:] >= 0)+start]
        
        
@dataclass
class SecondDiffZeroOverallUncertainty(OverallUncertainty):
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    wait_iters: int = 5
        
    start_method: str = "1"
        
    start_threshold: float = -1e-1
    start_iters: int = 5
        
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        sec = np.array(no_ahead_tvregdiff(grad, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad, sec
    
    def determine_start(self, sec):
        # Find nth value where the threshold is exceeded
        if np.count_nonzero(sec <= self.start_threshold) < self.start_iters:
            raise FailedToTerminate('FirstDiffZeroOverallUncertainty')
        start = np.searchsorted(np.cumsum(sec <= self.start_threshold), self.start_iters)
    
    def condition(self, x, metric):
        met, grad, sec = metric
        
        if self.start_method == "1":
            start = self.determine_start(sec)
        else:
            start = self.threshold_determine_start(met)
            
        if not (sec[start:] >= 0).any():
            raise FailedToTerminate('FirstDiffZeroOverallUncertainty')
        
        # stop when we hit zero
        return x.iloc[np.argmax(sec[start:] >= 0)+start]


@dataclass
class ClassificationChange(Criteria):
    """
    Stop if the predictions on the unlabelled pool does not change between two rounds.

    https://www.aclweb.org/anthology/C08-1142.pdf
    """

    @listify
    def metric(self, classifiers, **kwargs):
        yield np.nan

        for i in range(1, len(classifiers)):
            yield np.count_nonzero(
                classifiers[i - 1].predict(classifiers[i].X_unlabelled)
                == classifiers[i].predict(classifiers[i].X_unlabelled)
            ) / classifiers[i].X_unlabelled.shape[0]

    def condition(self, x, metric):
        if not any(np.isclose(x, 1) for x in metric):
            raise FailedToTerminate("classification_change")

        return x.iloc[np.argmax(np.isclose(metric[1:], 1))+1]


def reconstruct_unlabelled(clfs, X_unlabelled, Y_oracle, dense_atol=1e-6):
    """
    Reconstruct the unlabelled pool from stored information. We do not directly store the unlabelled pool,
    but we store enough information to reproduce it. This was used to compute stopping conditions implemented
    after some experiments had been started.
    """
    # Defensive asserts
    assert clfs is not None, "Classifiers must be non-none"
    assert X_unlabelled is not None, "X_unlabelled must be non-none"
    assert Y_oracle is not None, "Y_oracle must be non-none"
    assert (
        X_unlabelled.shape[0] == Y_oracle.shape[0]
    ), "unlabelled and oracle pools have a different shape"

    # TODO: This check is probably a bit slow, instead we could just check if the store is
    # version 2?
    if all(hasattr(clf, "X_unlabelled") for clf in clfs):
        yield from (clf.X_unlabelled for clf in clfs)

    # Fast row-wise compare function
    def compare(A, B, sparse):
        "https://stackoverflow.com/questions/23124403/how-to-compare-2-sparse-matrix-stored-using-scikit-learn-library-load-svmlight-f"
        if sparse:
            pairs = np.where(
                np.isclose(
                    (np.array(A.multiply(A).sum(1)) + np.array(B.multiply(B).sum(1)).T)
                    - 2 * A.dot(B.T).toarray(),
                    0,
                )
            )
            # TODO: Assert A[A_idx] == B[B_idx] for all pairs? Harder with sparse matrices.
        else:
            dists = euclidean_distances(A, B, squared=True)
            pairs = np.where(np.isclose(dists, 0, atol=dense_atol))
            for A_idx, B_idx in zip(*pairs):
                try:
                    assert (A[A_idx] == B[B_idx]).all()
                except AssertionError as e:
                    print(A[A_idx])
                    print(B[A_idx])
                    raise e
        return pairs

    # Yield the initial unlabelled pool
    yield X_unlabelled.copy()

    for clf in clfs[1:]:
        assert X_unlabelled.shape[0] == Y_oracle.shape[0]

        # make sure we're only checking for values from the 10 most recently added points
        # otherwise we might think a duplicate is a new point and try to add it, making the
        # count wrong!
        equal_rows = list(
            compare(
                X_unlabelled,
                clf.X_training[-10:],
                sparse=isinstance(X_unlabelled, scipy.sparse.csr_matrix),
            )
        )
        # Index fixing?
        equal_rows[1] = equal_rows[1] + (clf.X_training.shape[0] - 10)

        # Some datasets (rcv1) contain duplicates. These were only queried once, so we
        # make sure we only remove a single copy from the unlabelled pool.
        if len(equal_rows[0]) > 10:
            logger.debug(f"Found {len(equal_rows[0])} equal rows")
            really_equal_rows = []
            for clf_idx in np.unique(equal_rows[1]):
                dupes = equal_rows[0][equal_rows[1] == clf_idx]
                # some datasets have duplicates with differing labels (rcv1)
                dupes_correct_label = dupes[
                    (Y_oracle[dupes] == clf.y_training[clf_idx])
                    &
                    # this check is necessary so we don't mark an instance for removal twice
                    # when we want to mark another duplicate
                    np.logical_not(np.isin(dupes, really_equal_rows))
                ][0]
                really_equal_rows.append(dupes_correct_label)
            logger.debug(f"Found {len(really_equal_rows)} really equal rows")

        elif len(equal_rows[0]) == 10:
            # Fast path with no duplicates
            assert (Y_oracle[equal_rows[0]] == clf.y_training[equal_rows[1]]).all()
            really_equal_rows = equal_rows[0]
        else:
            raise Exception(
                f"Less than 10 ({len(equal_rows[0])}) equal rows were found."
                + " This could indicate an issue with the row-wise compare"
                + " function."
            )

        assert len(really_equal_rows) == 10, f"{len(really_equal_rows)}==10"

        n_before = X_unlabelled.shape[0]
        if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
            X_unlabelled = delete_from_csr(X_unlabelled, really_equal_rows)
        else:
            X_unlabelled = np.delete(X_unlabelled, really_equal_rows, axis=0)
        Y_oracle = np.delete(Y_oracle, really_equal_rows, axis=0)
        assert (
            X_unlabelled.shape[0] == n_before - 10
        ), f"We found 10 equal rows but {n_before-X_unlabelled.shape[0]} were removed"

        yield X_unlabelled.copy()


class NSupport(Criteria):
    """
    Determine a stopping point based on when the number of support vectors saturates.

    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.6090&rep=rep1&type=pdf
    """

    threshold: float = 0
    stable_iters: int = 2

    def metric(self, classifiers, n_support, **kwargs):
        if any(getattr(clf.estimator, "kernel", None) != "linear" for clf in classifiers):
            raise InvalidAssumption("n_support only supports linear SVMs")
        return n_support

    def condition(self, x, metric):
        return x.iloc[__is_approx_constant(metric, threshold=self.threshold, stable_iters=self.stable_iters)]


@dataclass
class StabilizingPredictions(Criteria):
    """
    Determine a stopping point based on agreement between past classifiers on a stopping
    set.

    This implementation uses a subsampled pool of 1000 instances, like all other methods we
    evaluate, as supposed to the 2000 suggested in the paper.

    * `k` determines how many classifiers are checked for agreement
    * `threshold` determines the kappa level that must be reached to halt

    https://arxiv.org/pdf/1409.5165.pdf
    """

    threshold: float = 0.99
    k: int = 3
    weak_threshold: float = 0.9

    def metric(self, x, classifiers, **kwargs):
        return kappa_metric(x, classifiers, k=self.k)
    
    def weak_determine_start(self, metric):
        return np.argmax(metric >= self.weak_threshold)

    def condition(self, x, metric):
        if not (metric >= self.threshold).any():
            raise FailedToTerminate("stabilizing_predictions")
        return x[np.argmax(metric >= self.threshold)]
    
    
@dataclass
class StabilizingPredictionsPlusX(Criteria):
    add_x: int = 100
    
    def condition(self, x, metric):
        return super().condition(x, metric) + self.add_x
    
    
@dataclass
class FirstDiffZeroStabilizingPredictions(StabilizingPredictions):
    alpha: float = 1e-1
    diffkernel: str = 'sq'
    start_method: str = "1"
    
    def metric(self, **kwargs):
        met = super().metric(**kwargs)
        grad = np.array(no_ahead_tvregdiff(met, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        return met, grad
    
    def determine_start(self, grad):
        # Find first point > 0; TODO: This might need to be more rigorous.
        return np.argmax(grad>0)
    
    def condition(self, x, metric):
        met, grad = metric
        
        if self.start_method == "1":
            start = self.determine_start(grad)
        else:
            start = self.weak_determine_start(met)
        
        # Find first point after that <= 0
        if not (grad[start:]<=0).any():
            raise FailedToTerminate("FirstDiffZeroStabilizingPredictions")
        
        return x.iloc[np.argmax(grad[start:]<=0)]
    

@listify
def metric_selected(classifiers, metric, aggregator=np.min, **kwargs):
    """
    Generator that produces the values of `metric` evaluated on the selected instances in each round of AL.

    The metric is then aggregated across the batch by the `aggregator` to produce a single value per round.
    """
    
    for i in range(1, len(classifiers)):
        yield aggregator(
            metric(classifiers[i - 1].estimator, classifiers[i].X_training[-10:])
        )
        
    yield np.nan


@dataclass
class MaxConfidence(Criteria):
    """
    This strategy is based on uncertainty measurement, considering whether the entropy of each selected unlabelled
    example is less than a very small predefined threshold close to zero, such as 0.001.

    Note: The original authors only considered non-batch mode AL. We stop based on the min of the entropy.

    https://www.aclweb.org/anthology/D07-1082.pdf
    """

    threshold: float = 0.001

    def metric(self, classifiers, **kwargs):
        return np.array(metric_selected(classifiers, classifier_entropy, aggregator=np.min))


    def condition(self, x, metric):
        if not (metric < self.threshold).any():
            raise FailedToTerminate("max_confidence")
            
        # We don't do the last selection so have one NaN at the end of the metric
        return x[np.argmax(metric[:-1] < self.threshold)]


class GOAL(Criteria):
    """
    Determine a stopping point based on the accuracy of previously trained classifiers.
    """

    safety_factor: float = 1.0
    order: int = 2
    # Addition from the original paper, sq works better in most scenarios (TODO: try it!)
    diffkernel: str = 'abs'
    # Affects regularization, 1e-1 or 1e-2 are best
    alpha: float = 1e-1


    def metric(self, classifiers, **kwargs):
        assert self.safety_factor >= 1.0, "safety factor cannot be less than 1"
        accx, _acc = first_acc(classifiers)
        grad = np.array(no_ahead_tvregdiff(_acc, 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False))
        start = np.argmax(grad < 0)

        if self.order == 2:
            second = np.array(
                [
                    np.nan,
                    np.nan,
                    *no_ahead_tvregdiff(grad[2:], 1, self.alpha, diffkernel=self.diffkernel, plotflag=False, diagflag=False),
                ]
            )

        return (grad, start, second, accx)

    def condition(self, x, metric):
        # This is hacky, not sure if there's a better way...
        grad, start, second, accx = metric
        if self.order == 2:
            try:
                return accx[
                    int(
                        (np.argmax((grad[start:] >= 0) & (second[start:] >= 0)) + start)
                        * self.safety_factor
                    )
                ]
            except IndexError:
                raise FailedToTerminate("ZPS")
        else:
            try:
                return accx[int((np.argmax(grad[start:] >= 0) + start) * self.safety_factor)]
            except IndexError:
                raise FailedToTerminate("ZPS")


# ----------------------------------------------------------------------------------------------
# Metric calculations
# ----------------------------------------------------------------------------------------------


def acc_last(classifiers, metric=metrics.accuracy_score):
    """
    Calculate the accuracy of earlier classifiers on the current dataset.
    """
    x = []
    diffs = []
    start = 1
    
    pclf = classifiers[0]
    unique_labels = np.unique(pclf.y_training)
    for i in range(start, len(classifiers)):
        clf = classifiers[i]
        x.append(clf.X_training.shape[0])

        if clf.y_training.shape[0] <= 10:
            diffs.append(np.inf)
            continue
        size = 10

        diffs.append(
            metric(clf.y_training[-size:], pclf.predict(clf.X_training[-size:]))
        )
        pclf = clf
        gc.collect()

    return x, diffs



def acc(classifiers, metric=metrics.accuracy_score, nth=0):
    """
    Calculate the accuracy of earlier classifiers on the current dataset.
    """
    x = []
    diffs = []
    if isinstance(nth, int):
        start = nth + 1
    elif nth == "last":
        start = 1
    else:
        start = 0
    for i, clf in enumerate(classifiers[start:]):
        x.append(clf.X_training.shape[0])

        if clf.y_training.shape[0] <= 10:
            diffs.append(np.inf)
            continue
        size = 10
        if isinstance(nth, int):
            pclf = classifiers[nth]
        elif nth == "last":
            pclf = classifiers[i]
            assert pclf.y_training.shape[0] < clf.y_training.shape[0]
        else:
            pclf = classifiers[0]

        unique_labels = np.unique(clf.y_training)
        if metric == roc_auc_score:
            if len(unique_labels) > 2 or len(clf.y_training.shape) > 1:
                diffs.append(
                    metric(
                        clf.y_training[-size:],
                        pclf.predict_proba(clf.X_training[-size:]),
                        multi_class="ovr",
                    )
                )
            else:
                diffs.append(
                    metric(
                        clf.y_training[-size:],
                        pclf.predict_proba(clf.X_training[-size:])[:, 1],
                    )
                )
        elif metric == f1_score:
            diffs.append(
                metric(
                    clf.y_training[-size:],
                    pclf.predict(clf.X_training[-size:]),
                    average="micro" if len(unique_labels) > 2 else "binary",
                    pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1,
                )
            )

        else:
            diffs.append(
                metric(clf.y_training[-size:], pclf.predict(clf.X_training[-size:]))
            )
        gc.collect()
    return x, diffs


def first_acc(classifiers, metric=metrics.accuracy_score, **kwargs):
    """
    Calculate the accuracy of first classifier on current data.

    Simplified for sanity checking.
    """

    x = []
    diffs = []

    for i, clf in enumerate(classifiers[1:]):
        x.append(clf.X_training.shape[0])

        if clf.y_training.shape[0] <= 10:
            diffs.append(np.inf)
            continue
        size = 10

        pclf = classifiers[0]

        unique_labels = np.unique(clf.y_training)
        if metric == roc_auc_score:
            prediction = pclf.predict_proba(clf.X_training)
            try:
                if len(unique_labels) > 2 or len(clf.y_training.shape) > 1:
                    diffs.append(metric(clf.y_training, prediction, multi_class="ovr"))
                else:
                    diffs.append(metric(clf.y_training, prediction[:, 1]))
            except ValueError:
                print(prediction)
        elif metric == f1_score:
            diffs.append(
                metric(
                    clf.y_training[-size:],
                    pclf.predict(clf.X_training[-size:]),
                    average="micro" if len(unique_labels) > 2 else "binary",
                    pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1,
                )
            )
        else:
            diffs.append(metric(clf.y_training, pclf.predict(clf.X_training)))

    return x, diffs



def kappa_metric(x, classifiers, k=3, **kwargs):
    from sklearn.preprocessing import OneHotEncoder

    out = [np.nan] * (k - 1)
    X_unlabelled = classifiers[0].X_unlabelled

    # Authors used 2000, but probably better to go with our standard 1000
    X_subsampled = X_unlabelled[
        np.random.choice(X_unlabelled.shape[0], 1000, replace=False)
    ]
    predictions = np.array([clf.predict(X_subsampled) for clf in classifiers])

    for i in range(k - 1, len(classifiers)):
        # Average the pairwise kappa score over the last k classifiers.
        kappa = np.mean(
            [
                cohen_kappa_score(x, y)
                for x, y in itertools.combinations(predictions[i - k + 1 : i + 1], 2)
            ]
        )

        out.append(kappa)

    return np.array(out)


def no_ahead_tvregdiff(value, *args, **kwargs):
    # Filter out leading NaNs before running TVRegDiff, it propagates but we want to ignore (but keep in output)
    first_nonnan = np.argmax(~np.isnan(value))
    
    # The first two values are nan regardless as we need to evaluate on shape of at least 2
    out = [np.nan]*(2+first_nonnan)
    for i in range(first_nonnan+2, len(value)):
        out.append(TVRegDiff(value[first_nonnan:i], *args, **kwargs)[-1])
    return out


# ----------------------------------------------------------------------------------------------
# Metric evaluators
# ----------------------------------------------------------------------------------------------

def __is_approx_constant(values, stable_iters=3, threshold=1e-1, **kwargs):
    """
    Determine if the input is approximately constant
    """

    const = values[0]
    n = 0
    for i, value in enumerate(values[1:]):
        if n == stable_iters:
            # TODO: Should this be i, i-1?
            return i + 1
        if np.abs(value - const) <= threshold:
            n += 1
        else:
            const = value
            n = 0
    return -1

# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


StopResult = namedtuple('StopResult', ['instances', 'accuracy_score', 'f1_score', 'roc_auc_score', 'metric'])


def eval_cond(dataset_results: Dict, name: str, conf: Config, condcls: Type, j: int, recompute: list, memory_profile: bool = False, **kwargs):
    # Restore saved result if possible
    if (
        name in dataset_results
        and len(dataset_results[name]) > j
        and name not in recompute
    ):
        print(f"Restoring saved result for {j}th run of {name} on {conf.dataset_name}")
        return dataset_results[name][j][0], dataset_results[name][j][4]

    # Memory profiling
    if memory_profile:
        from pympler import asizeof
        print(asizeof.asized(locals(), detail=3).format())
            
    metric_start = time.monotonic()

    # Instantiate condition class
    cond = condcls()

    # Attempt to evaluate the metric
    try:
        metric = cond.metric(**kwargs)
    except InvalidAssumption:
        return None, None
    except Exception as e:
        print(
            f"WARNING {name} failed evaluating metric on {conf.dataset_name} run {j} with exception: {e}"
        )
        raise e
        
    print(f"Evaluating metric {name} on {conf.dataset_name} took:", str(datetime.timedelta(seconds=time.monotonic()-metric_start)))
    cond_start = time.monotonic()

    # Attempt to evaluate, and return, the stop point & metric
    try:
        return cond.condition(x=kwargs['x'], metric=metric), metric
    except FailedToTerminate:
        print(f"{name} failed to terminate on {conf.dataset_name} run {j}")
        return None, metric
    except Exception as e:
        print(
            f"WARNING {name} failed evaluating metric on {conf.dataset_name} run {j} with exception: {e}"
        )
        raise e
    finally:
        print(f"Evaluating cond {name} on {conf.dataset_name} took:", str(datetime.timedelta(seconds=time.monotonic()-cond_start)))
        print(f"Evaluating {name} on {conf.dataset_name} took:", str(datetime.timedelta(seconds=time.monotonic()-metric_start)))


def eval_stopping_conditions(results_plots, classifiers, conditions=None, recompute=[], jobs=None, save=True, memory_profile=False):
    if conditions is None:
        conditions = {
            f"{f.display_name}": f for f in Criteria.all_criteria()
        }

    stop_results = {}

    if memory_profile:
        from pympler import asizeof
        print(asizeof.asized(locals(), detail=3).format())

    for (clfs, (conf, metrics)) in zip(classifiers, results_plots):
        print(f"Starting {conf.model_name} {conf.dataset_name}")
        stop_results[conf.dataset_name] = __read_stopping(conf.serialize())

        jobs = jobs if jobs is not None else min(os.cpu_count(), len(metrics) * len(conditions))
        results = np.array(
            Parallel(n_jobs=jobs)(
                delayed(eval_cond)(
                    stop_results[conf.dataset_name],
                    name,
                    conf,
                    cond,
                    j,
                    recompute,
                    **metric,
                    classifiers=clfs_,
                    config=conf,
                    memory_profile=memory_profile
                )
                for j, clfs_, metric in zip(conf.runs, clfs, metrics)
                for (name, cond) in conditions.items()
            ), 
            dtype=object
        ).reshape(len(metrics), len(conditions), 2)

        for i in range(len(conditions)):
            try:
                assert len(conf.runs) == len(results[:,i])
                stop_results[conf.dataset_name][list(conditions.keys())[i]] = {
                    runid: StopResult(
                        x
                            if list(conditions.keys())[i] != "SSNCut"
                            else (x + 10 if x is not None else None),
                        metrics[j]["accuracy_score"][metrics[j].x == x].iloc[0]
                            if x is not None
                            else None,
                        metrics[j]["f1_score"][metrics[j].x == x].iloc[0]
                            if x is not None
                            else None,
                        metrics[j]["roc_auc_score"][metrics[j].x == x].iloc[0]
                            if x is not None
                            else None,
                        metric
                    )
                    for runid, (j, (x, metric)) in zip(conf.runs, enumerate(results[:, i]))
                }
            except IndexError as e:
                print(
                    f"condition {list(conditions.keys())[i]} returned on dataset {conf.dataset_name}:\n{results[:,i]}"
                )
                raise e
        if save:
            print(f"Saving stop results to {conf.serialize()}")
            __write_stopping(conf.serialize(), stop_results[conf.dataset_name])

    return (conditions, stop_results)


def __read_stopping(config_str):
    file = f"{out_dir()}/stopping2/{config_str}.pickle"
    try:
        with open(file, "rb") as f:
            obj = dill.load(f)
            if type(obj) is list:
                obj = {i: v for i, v in enumerate(obj)}

            return obj
    except FileNotFoundError:
        return dict()


def __write_stopping(config_str, data):
    file = f"{out_dir()}/stopping2/{config_str}.pickle"
    with open(file, "wb") as f:
        return dill.dump(data, f)


# ----------------------------------------------------------------------------------------------
# Summary stats
# ----------------------------------------------------------------------------------------------


def summary(values):
    return f"{np.min(values):.0f}, {np.median(values):.0f}, {np.max(values):.0f}"


def stopped_at(stop_results):
    print(
        tabulate(
            [
                [
                    dataset,
                    *[
                        summary(stop_results[dataset][method])
                        for method in stop_results[dataset].keys()
                    ],
                ]
                for dataset in stop_results.keys()
            ],
            headers=stop_results["bbbp"].keys(),
            tablefmt="fancy_grid",
        )
    )


def optimal_dist(stop_results, optimal="optimal_fixed"):
    results = [
        [
            dataset,
            *[
                summary(
                    np.array(stop_results[dataset][method])
                    - stop_results[dataset][optimal]
                )
                for method in stop_results[dataset].keys()
                if not method.startswith(optimal)
            ],
        ]
        for dataset in stop_results.keys()
    ]
    diffs = np.array(
        [
            [
                np.array(stop_results[dataset][method]) - stop_results[dataset][optimal]
                for method in stop_results[dataset].keys()
                if not method.startswith(optimal)
            ]
            for dataset in stop_results.keys()
        ]
    )
    results.append(
        [
            "Total",
            *[
                f"{np.min(diffs[:,method])}, {np.mean(np.abs(np.median(diffs[:,method], axis=1)), axis=0)}, {np.max(diffs[:,method])}"
                for method in range(diffs.shape[1])
            ],
        ]
    )

    print(
        tabulate(
            results,
            headers=[
                k for k in stop_results["bbbp"].keys() if not k.startswith(optimal)
            ],
            tablefmt="fancy_grid",
        )
    )


def performance(stop_results, results, metric="roc_auc_score", optimal="optimal_fixed"):
    """
    Returns the performance of the stopped classifiers **as a fraction of the maximum achieved performance**.
    """
    results = [
        [
            dataset,
            *[
                f"{np.median([results[i][1][metric][results[i][1].x == run]/np.max(results[i][1][metric]) for run in stop_results[dataset][method]]):.0%}"
                for method in stop_results[dataset].keys()
            ],
        ]
        for i, dataset in enumerate(stop_results.keys())
    ]
    print(
        tabulate(
            results,
            headers=[k for k in stop_results["bbbp"].keys()],
            tablefmt="fancy_grid",
        )
    )


def in_bounds(stop_results):
    runs = len(next(iter(next(iter(stop_results.values())).values())))
    print(
        tabulate(
            [
                [
                    dataset,
                    *[
                        np.count_nonzero(
                            np.logical_and(
                                np.array(stop_results[dataset][method])
                                <= stop_results[dataset]["optimal_ub"],
                                np.array(stop_results[dataset][method])
                                >= stop_results[dataset]["optimal_lb"],
                            )
                        )
                        / runs
                        for method in stop_results[dataset].keys()
                        if not method.startswith("optimal")
                    ],
                ]
                for dataset in stop_results.keys()
            ],
            headers=[
                k for k in stop_results["bbbp"].keys() if not k.startswith("optimal")
            ],
            tablefmt="fancy_grid",
        )
    )


def rank_stop_conds(
    stop_results,
    metric,
    ax=None,
    title=None,
    average=False,
    passive=False,
    func=None
):
    data = []
    # n instances data
    for i, dataset in enumerate(stop_results.keys()):
        
        max_inst = 0
        min_acc = 1.
        for rs in stop_results[dataset].values():
            for r in rs:
                if r[0] is not None:
                    max_inst = max(max_inst, r[0])
                    min_acc = min(min_acc, r[1])
        
        for ii, method in enumerate(stop_results[dataset].keys()):
            if i == 0:
                data.append([])
            values = []
            for iii, (x, accuracy, f1, roc_auc, *_metric) in enumerate(
                stop_results[dataset][method]
            ):
                if x is None and metric != "func":
                    # TODO: Decide what to do with missing observations
                    values.append(None)
                elif metric == "instances":
                    values.append(x)
                elif metric == "func":
                    if x is not None:
                        values.append(
                            func(x, accuracy, f1, roc_auc)
                        )
                    else:
                        values.append(
                            func(max_inst, min_acc, None, None)
                        )
                else:
                    metrics = {"accuracy_score": 0, "f1_score": 1, "roc_auc_score": 2}
                    values.append((accuracy, f1, roc_auc)[metrics[metric]])
            if average:
                if np.count_nonzero(values) == 0:
                    # TODO: Replace -1e99 with passive values
                    data[ii].append(1e99 if metric == "instances" else -1e99)
                else:
                    data[ii].append(np.mean([v for v in values if v is not None]))
            else:
                for v in values:
                    penalty = 1e99 if metric == "instances" else -1e99
                    data[ii].append(v if v is not None else penalty)

    data = pd.DataFrame(
        np.array(data).T,
        columns=list(stop_results[list(stop_results.keys())[0]].keys()),
    )
    autoranked = autorank(
        data, order="ascending" if metric == "instances" or metric == "func" else "descending"
    )

    ax = plot_stats(autoranked, ax=ax)
    if ax is not None:
        ax.set_title(title or metric.rsplit("_score")[0].replace("_", " ").title())
    else:
        ax.figure.suptitle(
            title or metric.rsplit("_score")[0].replace("_", " ").title()
        )
    return ax
