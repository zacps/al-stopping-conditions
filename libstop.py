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

import time
import itertools
import os
import logging
from functools import partial
import operator

import dill
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
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


logger = logging.getLogger(__name__)


class FailedToTerminate(Exception):
    def __init__(self, method):
        super().__init__(f"{method} failed to determine a stopping location")


class InvalidAssumption(Exception):
    def __init__(self, method, reason):
        super().__init__(f"{method} could not be evaluated because: {reason}")


def SC_entropy_mcs(x, entropy_max, threshold=0.01, **kwargs):
    """
    Determine a stopping point based on when *all* samples in the unlabelled pool are
    below a threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """
    if not (entropy_max <= threshold).any():
        raise FailedToTerminate("SC_entropy_mcs")
    return x.iloc[np.argmax(entropy_max <= threshold)]


def SC_oracle_acc_mcs_values(classifiers, pre=None, **kwargs):
    """
    Determine a stopping point based on when the accuracy of the current classifier on the
    just-labelled samples is above a threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    return pre if pre is not None else acc(classifiers, nth="last")[1]


def SC_oracle_acc_mcs(x, classifiers, threshold=0.9, **kwargs):
    """
    Determine a stopping point based on when the accuracy of the current classifier on the
    just-labelled samples is above a threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    accx, acc_ = acc(classifiers, nth="last")
    try:
        return accx[np.argmax(np.array(acc_)[1:] >= threshold) + 1]
    except IndexError:
        raise FailedToTerminate("SC_oracle_acc_mcs")


def SC_mes(x, expected_error_min, threshold=1e-2, **kwargs):
    """
    Determine a stopping point based on the expected error of the classifier is below a
    threshold.

    https://www.aclweb.org/anthology/I08-1048.pdf
    """
    if (expected_error_min <= threshold).any():
        return x.iloc[np.argmax(expected_error_min <= threshold)]

    raise FailedToTerminate("SC_mes")


def EVM(
    x,
    uncertainty_variance,
    uncertainty_variance_selected,
    selected=True,
    n=2,
    m=1e-3,
    **kwargs,
):
    """
    Determine a stopping point based on the variance of the uncertainty in the unlabelled pool.

    * `selected` determines if the variance is calculated accross the entire pool or only the instances
    to be selected.

    * `n` the number of values for which the variance must decrease sequentially, paper used `2`.
    * `m` the threshold for which the variance must decrease by to be considered decreasing, paper
    used `0.5`.

    https://www.aclweb.org/anthology/W10-0101.pdf

    """
    variance = uncertainty_variance_selected if selected else uncertainty_variance
    current = 0
    last = variance[0]
    for i, value in enumerate(variance[1:]):
        if current == 2:
            # This used to be -1, which was a bug I think?
            return x[i + 1]
        if value < last - m:
            current += 1
        last = value
    raise FailedToTerminate("EVM")


def VM(
    x,
    uncertainty_variance,
    uncertainty_variance_selected,
    selected=True,
    n=2,
    m=1e-3,
    **kwargs,
):
    """
    Determine a stopping point based on the variance of the uncertainty in the unlabelled pool.

    * `selected` determines if the variance is calculated accross the entire pool or only the instances
    to be selected.

    * `n` the number of values for which the variance must decrease sequentially, paper used `2`.

    https://www.aclweb.org/anthology/W10-0101.pdf
    """
    return EVM(
        x,
        uncertainty_variance,
        uncertainty_variance_selected,
        selected=selected,
        n=n,
        m=0,
    )


def SSNCut(
    x, classifiers, X_unlabelled, Y_oracle, pre=None, m=0.2, affinity="linear", **kwargs
):
    if len(np.unique(classifiers[0].y_training)) > 2:
        raise InvalidAssumption("SSNCut", "dataset was not binary")
    values = (
        pre
        if pre is not None
        else SSNCut_values(
            classifiers, X_unlabelled, Y_oracle, m=0.2, affinity="linear"
        )
    )

    smallest = np.infty
    x0 = 0
    for i, v in enumerate(values):
        if v < smallest:
            smallest = v
            x0 = 0
        else:
            x0 += 1
        if x0 == 10:
            return x.iloc[i - 10]  # return to the point of lowest value

    raise FailedToTerminate("SSNCut")


def SSNCut_values(
    classifiers,
    X_unlabelled,
    Y_oracle,
    m=0.2,
    affinity="linear",
    dense_atol=1e-6,
    **kwargs,
):
    """
    NOTES:
    * They used an RBF svm to match the RBF affinity measure for SpectralClustering
    * As we carry out experiments on a linear svm we also use a linear affinity matrix (by default)

    file:///F:/Documents/Zotero/storage/DJGRDXSK/Fu%20and%20Yang%20-%202015%20-%20Low%20density%20separation%20as%20a%20stopping%20criterion%20for.pdf
    """

    if all(getattr(clf.estimator, "kernel", None) != "linear" for clf in classifiers):
        raise InvalidAssumption("SSNCut", "model is not a linear SVM")

    unique_y = np.unique(classifiers[0].y_training)
    if len(unique_y) > 2:
        print("WARNING: SSNCut is not designed for non-binary classification")

    clustering = SpectralClustering(n_clusters=unique_y.shape[0], affinity=affinity)

    out = []
    # Reconstruct the unlabelled pool if this run didn't save the unlabelled pool
    if hasattr(classifiers[0], "X_unlabelled"):
        # Don't store all pools in memory, generator expression
        X_unlabelleds = (clf.X_unlabelled for clf in classifiers)
    else:
        X_unlabelleds = reconstruct_unlabelled(
            classifiers, X_unlabelled, Y_oracle, dense_atol=dense_atol
        )

    for i, (clf, X_unlabelled) in enumerate(zip(classifiers, X_unlabelleds)):
        # print(f"{i}/{len(classifiers)}")
        t0 = time.monotonic()
        # Note: With non-binary classification the value of the decision function is a transformation of the distance...
        order = np.argsort(np.abs(clf.estimator.decision_function(X_unlabelled)))
        M = X_unlabelled[order[: min(1000, int(m * X_unlabelled.shape[0]))]]

        y0 = clf.predict(M)
        # use algorithm 1
        scipy.sparse.save_npz("M.npz", M)
        y1 = clustering.fit_predict(M)

        y0 = LabelEncoder().fit_transform(y0)
        diff = np.sum(y0 == y1) / X_unlabelled.shape[0]
        if diff > 0.5:
            diff = 1 - diff
        out.append(diff)
        # print(f"SSNCut took {time.monotonic()-t0}")

    return out


def fscore_tvdiff(x, classifiers, X_unlabelled, Y_oracle):
    """
    'Performance convergence' using TVDiff gradient estimation
    https://www.aclweb.org/anthology/C08-1059
    """
    values = list(fscore(classifiers, X_unlabelled, Y_oracle))
    grad = no_ahead_tvregdiff(values, 1, 1e-1, plotflag=False, diagflag=False)
    return x.iloc[2 + np.argmax(grad[2:] < 0)]


def contradictory_information(x, contradictory_information, rounds=3, **kwargs):
    """
    Stop when contradictory information drops for `rounds` consecutive rounds.

    https://www-sciencedirect-com.ezproxy.auckland.ac.nz/science/article/pii/S088523080700068X
    """

    current = 0
    last = contradictory_information[0]
    for i, value in enumerate(contradictory_information[1:]):
        if current == rounds:
            return x[i + 1]
        if value < last:
            current += 1
        last = value
    raise FailedToTerminate("contradictory_information")


def performance_convergence(
    x,
    classifiers,
    X_unlabelled,
    Y_oracle,
    pre=None,
    threshold=5e-5,
    k=10,
    average=np.mean,
    dense_atol=1e-6,
    **kwargs,
):
    """
    We use k=10 instead of 100 because batch size is 10 for us, 1 for them
    multiple thresholds tested by authors (1e-2, 5e-5)

    https://www.aclweb.org/anthology/C08-1059.pdf
    """

    perf = (
        pre
        if pre is not None
        else list(fscore(classifiers, X_unlabelled, Y_oracle, dense_atol=dense_atol))
    )
    windows = np.lib.stride_tricks.sliding_window_view(perf, k)
    for i in range(1, len(windows)):
        w2 = average(windows[i])
        w1 = average(windows[i - 1])
        g = w2 - w1
        if windows[i][-1] > np.max(perf[: i + k]) and g > 0 and g < threshold:
            return x.iloc[i + k]

    raise FailedToTerminate("performance_convergence")


def uncertainty_convergence(
    x,
    classifiers,
    X_unlabelled,
    Y_oracle,
    pre=None,
    metric=classifier_entropy,
    aggregator=np.min,
    threshold=5e-5,
    k=10,
    average=np.median,
    **kwargs,
):
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

    uncertainty = pre if pre is not None else metric_selected(classifiers, metric)
    windows = np.lib.stride_tricks.sliding_window_view(uncertainty, k)
    for i in range(1, len(windows)):
        w2 = average(windows[i])
        w1 = average(windows[i - 1])
        g = w2 - w1
        if windows[i][-1] > np.max(uncertainty[: i + k]) and g > 0 and g < threshold:
            return x.iloc[i + k]

    raise FailedToTerminate("performance_convergence")


def overall_uncertainty(x, uncertainty_average, **kwargs):
    """
    Stop if the overall uncertainty on the unlabelled pool is less than a threshold.

    https://www.aclweb.org/anthology/C08-1142.pdf
    """
    if not (uncertainty_average < 1e-2).any():
        raise FailedToTerminate("overall_uncertainty")
    return x.iloc[np.argmax(uncertainty_average < 1e-2)]


def classification_change(
    x, classifiers, X_unlabelled, Y_oracle, pre=None, dense_atol=1e-6, **kwargs
):
    """
    Stop if the predictions on the unlabelled pool does not change between two rounds.

    https://www.aclweb.org/anthology/C08-1142.pdf
    """

    values = (
        pre
        if pre is not None
        else classification_change_values(
            classifiers, X_unlabelled, Y_oracle, dense_atol=dense_atol
        )
    )

    if not any(x == 1 for x in values):
        raise FailedToTerminate("classification_change")

    print(np.argmax(values == 1))
    return x.iloc[np.argmax(values == 1)]


@listify
def classification_change_values(
    classifiers, X_unlabelled, Y_oracle, dense_atol=1e-6, **kwargs
):
    X_unlabelleds = reconstruct_unlabelled(
        classifiers, X_unlabelled, Y_oracle, dense_atol=dense_atol
    )

    # Skip the first pool
    next(X_unlabelleds)

    yield np.nan

    for i, X_unlabelled in zip(range(1, len(classifiers)), X_unlabelleds):
        # print(f"At iteration {i} unlabelled pool has shape {X_unlabelled.shape if X_unlabelled is not None else 'None'}")
        yield np.count_nonzero(
            classifiers[i - 1].predict(X_unlabelled)
            == classifiers[i].predict(X_unlabelled)
        ) / X_unlabelled.shape[0]


def fscore(classifiers, X_unlabelled, Y_oracle, dense_atol=1e-6, **kwargs):
    """
    https://www.aclweb.org/anthology/C08-1059
    """
    X_unlabelleds = reconstruct_unlabelled(
        classifiers, X_unlabelled, Y_oracle, dense_atol=dense_atol
    )

    for clf, X_unlabelled in zip(classifiers, X_unlabelleds):

        X_subsampled = X_unlabelled[
            np.random.choice(
                X_unlabelled.shape[0], min(X_unlabelled.shape[0], 1000), replace=False
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


def n_support(x, classifiers, n_support, **kwargs):
    """
    Determine a stopping point based on when the number of support vectors saturates.

    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.6090&rep=rep1&type=pdf
    """
    if all(getattr(clf.estimator, "kernel", None) != "linear" for clf in classifiers):
        print("WARN: n_support only supports linear SVMs")
        return x.iloc[-1]

    kwargs.setdefault("threshold", 0)
    kwargs.setdefault("stable_iters", 2)
    return x.iloc[__is_approx_constant(n_support, **kwargs)]


def stabilizing_predictions(
    x, classifiers, X_unlabelled, k=3, threshold=0.99, **kwargs
):
    """
    Determine a stopping point based on agreement between past classifiers on a stopping
    set.

    This implementation uses a subsampled pool of 1000 instances, like all other methods we
    evaluate, as supposed to the 2000 suggested in the paper.

    * `k` determines how many classifiers are checked for agreement
    * `threshold` determines the kappa level that must be reached to halt

    https://arxiv.org/pdf/1409.5165.pdf
    """
    metric = kappa_metric(x, classifiers, X_unlabelled, k=k)
    if not (metric >= threshold).any():
        raise FailedToTerminate("stabilizing_predictions")
    return x[np.argmax(metric >= threshold)]


def KD(x, classifiers, **kwargs):
    """
    Determine a stopping point based on the similarity of linear SVM hyperplanes.

    https://math.stackexchange.com/questions/2124611/on-a-measure-of-similarity-between-two-hyperplanes
    \|w_1\|\|w_2\|-|\langle w_1,w_2 \rangle| +|b_1-b_2|
    """
    if all(getattr(clf.estimator, "kernel", None) != "linear" for clf in classifiers):
        print("WARN: KD only supports linear SVMs")
        return x.iloc[-1]


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


def max_confidence(x, classifiers, pre=None, threshold=0.001, **kwargs):
    """
    This strategy is based on uncertainty measurement, considering whether the entropy of each selected unlabelled
    example is less than a very small predefined threshold close to zero, such as 0.001.

    Note: The original authors only considered non-batch mode AL. We stop based on the mean of the entropy.

    https://www.aclweb.org/anthology/D07-1082.pdf
    """

    metric = np.array(
        pre if pre is not None else metric_selected(classifiers, classifier_entropy)
    )
    if not (metric < threshold).any():
        raise FailedToTerminate("max_confidence")

    return x[np.argmax(metric < threshold)]


def uncertainty_min(x, uncertainty_min, **kwargs):
    """
    Stop if the minimum uncertainty in the unlabelled pool is less than a threshold.
    """
    kwargs.setdefault("stable_iters", 3)
    kwargs.setdefault("maximum", 1e-1)
    return x.iloc[__is_within_bound(uncertainty_min, **kwargs)]


def ZPS(classifiers, order=1, safety_factor=1.0, **kwargs):
    """
    Determine a stopping point based on the accuracy of previously trained classifiers.
    """
    assert safety_factor >= 1.0, "safety factor cannot be less than 1"
    accx, _acc = first_acc(classifiers)
    grad = np.array(no_ahead_tvregdiff(_acc, 1, 1e-1, plotflag=False, diagflag=False))
    start = np.argmax(grad < 0)

    if order == 2:
        second = np.array(
            [
                np.nan,
                np.nan,
                *no_ahead_tvregdiff(grad[2:], 1, 1e-1, plotflag=False, diagflag=False),
            ]
        )
        try:
            return accx[
                int(
                    (np.argmax((grad[start:] >= 0) & (second[start:] >= 0)) + start)
                    * safety_factor
                )
            ]
        except IndexError:
            raise FailedToTerminate("ZPS")

    try:
        return accx[int((np.argmax(grad[start:] >= 0) + start) * safety_factor)]
    except IndexError:
        raise FailedToTerminate("ZPS")


# ----------------------------------------------------------------------------------------------
# Metric calculations
# ----------------------------------------------------------------------------------------------


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


# Maybe viable but completely wrong
def kappa_metric_but_not_and_broken(x, classifiers, X_unlabelled, k=3, **kwargs):
    from sklearn.preprocessing import OneHotEncoder

    out = [np.nan, np.nan]
    classes = np.unique(classifiers[0].y_training)
    enc = LabelEncoder()
    enc.fit(classifiers[0].y_training)

    predictions = np.array(
        [enc.transform(clf.predict(X_unlabelled)) for clf in classifiers]
    )
    probabilities = []
    for preds in predictions:
        probabilities.append([])
        values = np.unique(preds, return_counts=True)
        for klass in enc.transform(classes):
            v = values[1][values[0] == klass]
            v = v[0] if v.shape[0] > 0 else 0
            probabilities[-1].append(v / X_unlabelled.shape[0])
        # for klass in enc.transform(classes):
        #    probabilities[-1].append(np.count_nonzero(preds==klass)/X_unlabelled.shape[0])
    probabilities = np.array(probabilities)
    print(probabilities)
    for i in range(2, len(classifiers)):
        k1 = np.sum(probabilities[i] * probabilities[i - 1])
        k2 = np.sum(probabilities[i] * probabilities[i - 2])
        k3 = np.sum(probabilities[i - 1] * probabilities[i - 2])
        assert k1 != np.nan and k2 != np.nan and k3 != np.nan
        kappa = np.mean([k1, k2, k3])

        out.append(kappa)

    return np.array(out)


def kappa_metric_first(x, classifiers, X_unlabelled, **kwargs):
    from sklearn.preprocessing import OneHotEncoder

    out = [np.nan]
    classes = np.unique(classifiers[0].y_training)

    X_subsampled = X_unlabelled[
        np.random.choice(X_unlabelled.shape[0], 1000, replace=False)
    ]

    predictions = np.array([clf.predict(X_subsampled) for clf in classifiers])

    for i in range(1, len(classifiers)):
        # Compute kappa against the first trained classifier
        kappa = cohen_kappa_score(predictions[0], predictions[i])

        out.append(kappa)

    return np.array(out)


def kappa_metric(x, classifiers, X_unlabelled, k=3, **kwargs):
    from sklearn.preprocessing import OneHotEncoder

    out = [np.nan] * (k - 1)
    classes = np.unique(classifiers[0].y_training)

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


def hyperplane_similarity(classifiers, nth="last"):
    """
    Returns shape (len(classifiers), (n_classes*(n_classes-1)//2))
    """
    from numpy.linalg import norm

    n_classes = classifiers[0].estimator.n_support_.shape[0]
    out = [np.full((n_classes * (n_classes - 1) // 2,), np.nan)]
    for i, clf in enumerate(classifiers[1:]):
        clf = clf.estimator
        pclf = classifiers[i if nth == "last" else 0].estimator

        #
        # 0 is perfectly similar
        out.append(
            [
                1
                - np.abs(np.inner(clf.coef_[i], pclf.coef_[i]))
                / (norm(clf.coef_[i]) * norm(clf.coef_[i]))
                + np.abs(clf.intercept_[i] - pclf.intercept_[i])
                / np.sqrt(np.abs(clf.intercept_[i] * pclf.intercept_[i]))
                for i in range(clf.coef_.shape[0])
            ]
        )
    return np.array(out)


def no_ahead_grad(x):
    """
    Compute a gradient across x, ensuring that no values ahead of the current are used for the
    gradient calculation.
    """
    return [np.inf, np.inf, *[np.gradient(x[:i])[-1] for i in range(2, len(x))]]


def no_ahead_tvregdiff(value, *args, **kwargs):
    out = [np.nan, np.nan]
    for i in range(2, len(value)):
        out.append(TVRegDiff(value[:i], *args, **kwargs)[-1])
    return out


# ----------------------------------------------------------------------------------------------
# Metric evaluators
# ----------------------------------------------------------------------------------------------


def __is_within_bound(values, minimum=None, maximum=None, stable_iters=3, **kwargs):
    n = 0
    for i, value in enumerate(values):
        if n == stable_iters:
            # TODO: Should this be i+1, i-1?
            return i
        if (minimum is None or value >= minimum) and (
            maximum is None or value <= maximum
        ):
            n += 1
        else:
            n = 0
    return -1


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


def __is_approx_minimum(value, stable_iters=3, threshold=0, **kwargs):
    """
    Determine if the input is approximately at a minimum by estimating the gradient.
    """

    pass


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


def eval_stopping_conditions(results_plots, classifiers, conditions=None):
    if conditions is None:
        params = {"kappa": {"k": 2}}
        conditions = {
            **{
                f"{f.__name__}": partial(f, **params.get(f.__name__, {}))
                for f in [
                    uncertainty_min,
                    SC_entropy_mcs,
                    SC_oracle_acc_mcs,
                    SC_mes,
                    EVM,
                    # ZPS_ee_grad,
                    stabilizing_predictions,
                ]
            },
            "ZPS2": partial(ZPS, order=2),
            # "SSNCut": SSNCut
        }

    stop_results = {}

    def eval_cond(name, conf, cond, j, **kwargs):
        if (
            name in stop_results[conf.dataset_name]
            and len(stop_results[conf.dataset_name][name]) > j
        ):
            if isinstance(stop_results[conf.dataset_name][name][j], list) or isinstance(
                stop_results[conf.dataset_name][name][j], tuple
            ):
                return stop_results[conf.dataset_name][name][j][0]
            else:
                return stop_results[conf.dataset_name][name][j]
        try:
            # Hack: Swarm needs a more relaxed tolerance for reconstruct_unlabelled
            if conf.dataset_name == "swarm":
                kwargs["dense_atol"] = 1e-1
            else:
                kwargs["dense_atol"] = 1e-6
            return cond(**kwargs)
        except FailedToTerminate:
            print(f"{name} failed to terminate on {conf.dataset_name} run {j}")
            return None
        except InvalidAssumption:
            return None
        except Exception as e:
            print(
                f"WARNING {name} failed on {conf.dataset_name} run {j} with exception: {e}"
            )
            raise e

    for (clfs, (conf, metrics)) in zip(classifiers, results_plots):
        stop_results[conf.dataset_name] = __read_stopping(conf.serialize())

        X, y = getattr(libdatasets, conf.dataset_name)(None)

        def pools(X, y, conf, runs):
            for i in runs:
                _, X_unlabelled, _, y_oracle, _, _ = active_split(
                    X,
                    y,
                    labeled_size=conf.meta["labelled_size"],
                    test_size=conf.meta["test_size"],
                    random_state=check_random_state(i),
                    ensure_y=conf.meta["ensure_y"],
                )
                assert X_unlabelled.shape[0] > 0, "Unlabelled pool cannot have length 0"
                assert (
                    y_oracle.shape[0] > 0
                ), "Unlabelled pool labels cannot have length 0"
                yield (X_unlabelled, y_oracle)

        it1, it2 = itertools.tee(pools(X, y, conf, conf.runs))
        unlabelled_pools = map(operator.itemgetter(0), it1)
        y_oracles = map(operator.itemgetter(1), it2)

        # todo: split this into chunks for memory usage reasons
        results = np.array(
            Parallel(n_jobs=min(os.cpu_count(), len(metrics) * len(conditions)))(
                delayed(eval_cond)(
                    name,
                    conf,
                    cond,
                    j,
                    **metric,
                    classifiers=clfs_,
                    config=conf,
                    X_unlabelled=X_unlabelled,
                    Y_oracle=y_oracle,
                )
                for j, (clfs_, metric, X_unlabelled, y_oracle) in enumerate(
                    zip(clfs, metrics, unlabelled_pools, y_oracles)
                )
                for (name, cond) in conditions.items()
            )
        ).reshape(len(metrics), len(conditions))

        for i in range(len(conditions)):
            try:
                stop_results[conf.dataset_name][list(conditions.keys())[i]] = [
                    (
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
                    )
                    for j, x in enumerate(results[:, i])
                ]
            except IndexError as e:
                print(
                    f"condition {list(conditions.keys())[i]} returned on dataset {conf.dataset_name}:\n{results[:,i]}"
                )
                raise e
        __write_stopping(conf.serialize(), stop_results[conf.dataset_name])

    return (conditions, stop_results)


def __read_stopping(config_str):
    file = f"{out_dir()}/stopping/{config_str}.pickle"
    try:
        with open(file, "rb") as f:
            return dill.load(f)
    except FileNotFoundError:
        return {}


def __write_stopping(config_str, data):
    file = f"{out_dir()}/stopping/{config_str}.pickle"
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
    results_plots,
    metric,
    ax=None,
    title=None,
    average=False,
    passive=False,
    holistic_x=50,
):
    data = []
    # n instances data
    for i, dataset in enumerate(stop_results.keys()):
        for ii, method in enumerate(stop_results[dataset].keys()):
            if i == 0:
                data.append([])
            values = []
            for iii, (x, accuracy, f1, roc_auc) in enumerate(
                stop_results[dataset][method]
            ):
                if x is None:
                    # TODO: Decide what to do with missing observations
                    values.append(None)
                elif metric == "instances":
                    values.append(x)
                elif metric == "holistic":
                    values.append((accuracy + roc_auc) / 2 * holistic_x * 100 - x)
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

    # print(len(data))
    # for x in data:
    #    print(f"  {len(x)}")
    data = pd.DataFrame(
        np.array(data).T,
        columns=list(stop_results[list(stop_results.keys())[0]].keys()),
    )
    autoranked = autorank(
        data, order="ascending" if metric == "instances" else "descending"
    )

    ax = plot_stats(autoranked, ax=ax)
    if ax is not None:
        ax.set_title(title or metric.rsplit("_score")[0].replace("_", " ").title())
    else:
        ax.figure.suptitle(
            title or metric.rsplit("_score")[0].replace("_", " ").title()
        )
    return ax
