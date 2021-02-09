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

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
from statsmodels.stats.inter_rater import fleiss_kappa

from tvregdiff.tvregdiff import TVRegDiff


def optimal_ub(x, accuracy_score, f1_score, roc_auc_score, **kwargs):
    # 150 samples is worth 1 percentage point of (accuracy/roc_auc)
    return optimal(x, accuracy_score, f1_score, roc_auc_score, **kwargs, aggressiveness=-275 * 100)

def optimal_lb(x, accuracy_score, f1_score, roc_auc_score, **kwargs):
    # 20 samples is worth 1 percentage point of (accuracy/roc_auc)
    return optimal(x, accuracy_score, f1_score, roc_auc_score, **kwargs, aggressiveness=-15 * 100)

def optimal(x, accuracy_score, f1_score, roc_auc_score, aggressiveness=-75*100, **kwargs):
    # 50 samples is worth 1 percentage point of (accuracy/roc_auc)
    return x[np.argmin(
        # Calculate the minimum, normalised metric. This gets double weight.
        np.sum([
            np.max([
                accuracy_score.to_numpy()*aggressiveness*np.max(accuracy_score), 
                f1_score.to_numpy()*aggressiveness*np.max(f1_score), 
                roc_auc_score.to_numpy()*aggressiveness*np.max(roc_auc_score)
            ], axis=0),
            accuracy_score*aggressiveness,
            f1_score*aggressiveness,
            roc_auc_score*aggressiveness,
        ],axis=0)/4 + x
    )]


def optimal_fixed(x, accuracy_score, f1_score, roc_auc_score, threshold=0.99, **kwargs):
    return x[np.argmax((accuracy_score>=np.max(accuracy_score)*threshold) & (roc_auc_score>=np.max(roc_auc_score)*threshold))] # & (f1_score>=np.max(f1_score)*threshold)


def SP(stop_set, **kwargs):
    pass


def SC_entropy_mcs(x, entropy_max, threshold=0.01, **kwargs):
    """
    Determine a stopping point based on when *all* samples in the unlabelled pool are
    below a threshold.
    
    https://www.aclweb.org/anthology/I08-1048.pdf
    """
    if not (entropy_max<=threshold).any():
        return x.iloc[-1]
    return x.iloc[np.argmax(entropy_max<=threshold)]


def SC_oracle_acc_mcs(x, classifiers, threshold=0.9, **kwargs):
    """
    Determine a stopping point based on when the accuracy of the current classifier on the 
    just-labelled samples is above a threshold.
    
    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    accx, acc_ = acc(classifiers, nth='last')
    return accx[np.argmax(np.array(acc_)[1:] >= threshold)+1]


def SC_mes(x, expected_error, threshold=0, **kwargs):
    """
    Determine a stopping point based on the expected error of the classifier is below a
    threshold.
    
    https://www.aclweb.org/anthology/I08-1048.pdf
    """

    return x[np.argmax(expected_error <= threshold)]


def EVM(x, uncertainty_variance, uncertainty_variance_selected, selected=True, n=2, m=1e-3, **kwargs):
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
            return x[i-1]
        if value < last-m:
            current += 1
        last = value
    return x.iloc[-1]

def n_support(x, classifiers, n_support, **kwargs):
    """
    Determine a stopping point based on when the number of support vectors saturates.
    
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.6090&rep=rep1&type=pdf
    """
    if all(getattr(clf.estimator, 'kernel', None) != 'linear' for clf in classifiers):
        print("WARN: n_support only supports linear SVMs")
        return x.iloc[-1]
    
    kwargs.setdefault('threshold', 0)
    kwargs.setdefault('stable_iters', 2)
    return x.iloc[__is_approx_constant(n_support, **kwargs)]


def kappa(x, classifiers, k=3, threshold=0.99, **kwargs):
    """
    Determine a stopping point based on agreement between past classifiers on a stopping
    set. 
    
    This implementation currently uses *all* non-validation data as the stop set, this 
    differs from the paper which used ~2000 instances from the pool.
    
    * `k` determines how many classifiers are checked for agreement 
    * `threshold` determines the kappa level that must be reached to halt
    
    https://arxiv.org/pdf/1409.5165.pdf
    """
    metric = kappa_metric(x, classifiers, k)
    if not (metric >=threshold).any():
        return x[-1]
    return x[np.argmax(metric>=threshold)]


def KD(x, classifiers, **kwargs):
    """
    Determine a stopping point based on the similarity of linear SVM hyperplanes.
    
    https://math.stackexchange.com/questions/2124611/on-a-measure-of-similarity-between-two-hyperplanes
    \|w_1\|\|w_2\|-|\langle w_1,w_2 \rangle| +|b_1-b_2|
    """
    if all(getattr(clf.estimator, 'kernel', None) != 'linear' for clf in classifiers):
        print("WARN: KD only supports linear SVMs")
        return x.iloc[-1]
    
    


def uncertainty_min(x, uncertainty_min, **kwargs):
    """
    Stop if the minimum uncertainty in the unlabelled pool is less than a threshold.
    """
    kwargs.setdefault('stable_iters', 3)
    kwargs.setdefault('maximum', 1e-1)
    return x.iloc[__is_within_bound(uncertainty_min, **kwargs)]


def ZPS(classifiers, **kwargs):
    """
    Determine a stopping point based on the accuracy of previously trained classifiers.
    """
    accx, _acc = first_acc(classifiers)
    grad = np.array(no_ahead_tvregdiff(_acc, 1, 1e-2, plotflag=False, diagflag=False))
    start = np.argmax(grad < 0)
    return accx[np.argmax(grad[start:] >= 0)+start]


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
        start = nth+1 
    elif nth == 'last':
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
        elif nth == 'last':
            pclf = classifiers[i]
            assert pclf.y_training.shape[0] < clf.y_training.shape[0]
        else:
            pclf = classifiers[0]
            
        if metric == roc_auc_score:
            if len(np.unique(clf.y_training)) > 2 or len(clf.y_training.shape) > 1:
                diffs.append(metric(clf.y_training[-size:], pclf.predict_proba(clf.X_training[-size:]), multi_class="ovr"))
            else:
                diffs.append(metric(clf.y_training[-size:], pclf.predict_proba(clf.X_training[-size:])[:,1]))
        else:
            diffs.append(metric(clf.y_training[-size:], pclf.predict(clf.X_training[-size:])))
    return x, diffs

def first_acc(classifiers, metric=metrics.accuracy_score):
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
        
        diffs.append(metric(clf.y_training, pclf.predict(clf.X_training)))
        
    return x, diffs


def kappa_metric(x, classifiers, k=3, **kwargs):
    from sklearn.preprocessing import OneHotEncoder
    
    out = [np.nan, np.nan]
    for i, clf in enumerate(classifiers[k-1:]):
        clfs = [*[classifiers[-i] for i in range(1, k)], clf]
        
        kappa = fleiss_kappa(np.sum([OneHotEncoder().fit_transform(clf.predict(classifiers[-1].X_training).reshape(-1, 1)).todense() for clf in clfs], axis=0))
        out.append(kappa)

    return np.array(out)


def hyperplane_similarity(classifiers, nth='last'):
    """
    
    Returns shape (len(classifiers), (n_classes*(n_classes-1)//2))
    """
    from numpy.linalg import norm
    n_classes = classifiers[0].estimator.n_support_.shape[0]
    out = [np.full((n_classes * (n_classes - 1) // 2,), np.nan)]
    for i, clf in enumerate(classifiers[1:]):
        clf = clf.estimator
        pclf = classifiers[i if nth == 'last' else 0].estimator
        
        # 
        # 0 is perfectly similar
        out.append([1-np.abs(np.inner(clf.coef_[i], pclf.coef_[i]))/(norm(clf.coef_[i])*norm(clf.coef_[i]))+np.abs(clf.intercept_[i] - pclf.intercept_[i])/np.sqrt(np.abs(clf.intercept_[i]*pclf.intercept_[i])) for i in range(clf.coef_.shape[0])])
    return np.array(out)


def no_ahead_grad(x):
    """
    Compute a gradient across x, ensuring that no values ahead of the current are used for the
    gradient calculation.
    """
    return [np.inf,np.inf, *[np.gradient(x[:i])[-1] for i in range(2, len(x))]]

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
        if (minimum is None or value >= minimum) and (maximum is None or value <= maximum):
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
            return i+1
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
# Summary stats
# ----------------------------------------------------------------------------------------------

def summary(values):
    return f"{np.min(values):.0f}, {np.median(values):.0f}, {np.max(values):.0f}"

def stopped_at(stop_results):
    print(tabulate(
        [
            [
                dataset,
                *[summary(stop_results[dataset][method]) for method in stop_results[dataset].keys()]
            ]
            for dataset in stop_results.keys()],
        headers=stop_results['bbbp'].keys(),
        tablefmt='fancy_grid'
    ))
    
def optimal_dist(stop_results, optimal='optimal_fixed'):
    results = [
        [
            dataset,
            *[summary(np.array(stop_results[dataset][method])-stop_results[dataset][optimal]) for method in stop_results[dataset].keys() if not method.startswith(optimal)]
        ]
        for dataset in stop_results.keys()
    ]
    diffs = np.array([[np.array(stop_results[dataset][method])-stop_results[dataset][optimal] for method in stop_results[dataset].keys() if not method.startswith(optimal)] for dataset in stop_results.keys()])
    results.append([
        "Total",
        *[f"{np.min(diffs[:,method])}, {np.mean(np.abs(np.median(diffs[:,method], axis=1)), axis=0)}, {np.max(diffs[:,method])}" for method in range(diffs.shape[1])]
    ])
    
    print(tabulate(
        results,
        headers=[k for k in stop_results['bbbp'].keys() if not k.startswith(optimal)],
        tablefmt='fancy_grid'
    ))
    
    
def performance(stop_results, results, metric='roc_auc_score', optimal='optimal_fixed'):
    """
    Returns the performance of the stopped classifiers **as a fraction of the maximum achieved performance**.
    """
    results = [
        [
            dataset,
            *[f"{np.median([results[i][1][metric][results[i][1].x == run]/np.max(results[i][1][metric]) for run in stop_results[dataset][method]]):.0%}" for method in stop_results[dataset].keys()]
        ]
        for i, dataset in enumerate(stop_results.keys())
    ]
    print(tabulate(
        results,
        headers=[k for k in stop_results['bbbp'].keys()],
        tablefmt='fancy_grid'
    ))
    
def in_bounds(stop_results):
    runs = len(next(iter(next(iter(stop_results.values())).values())))
    print(tabulate(
        [
            [
                dataset,
                *[np.count_nonzero(np.logical_and(
                    np.array(stop_results[dataset][method]) <= stop_results[dataset]['optimal_ub'], np.array(stop_results[dataset][method]) >= stop_results[dataset]['optimal_lb']
                ))/runs for method in stop_results[dataset].keys() if not method.startswith('optimal')]
            ]
            for dataset in stop_results.keys()],
        headers=[k for k in stop_results['bbbp'].keys() if not k.startswith('optimal')],
        tablefmt='fancy_grid'
    ))