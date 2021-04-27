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

from functools import partial
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.utils import check_random_state
from sklearn.metrics import roc_auc_score, f1_score
from libactive import active_split
from tabulate import tabulate
from statsmodels.stats.inter_rater import fleiss_kappa
from autorank import autorank, plot_stats

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


def SC_mes(x, expected_error_min, threshold=1e-2, **kwargs):
    """
    Determine a stopping point based on the expected error of the classifier is below a
    threshold.
    
    https://www.aclweb.org/anthology/I08-1048.pdf
    """
    
    return x.iloc[np.argmax(expected_error_min <= threshold) or -1]


def ZPS_ee(x, expected_error_min, threshold=5e-2, **kwargs):
    # WARN: This is slightly omniscient, but not in a way that should matter.
    # TODO: FIX IT!
    return x.iloc[np.argmax(expected_error_min <= threshold*np.max(expected_error_min))]


def ZPS_ee_grad(x, expected_error_min, threshold=10, **kwargs):
    grad = np.array(no_ahead_tvregdiff(expected_error_min[1:], 1, 1e-1, plotflag=False, diagflag=False))
    
    second = np.array([np.nan, np.nan, np.nan, *no_ahead_tvregdiff(grad[2:], 1, 15, plotflag=False, diagflag=False)])
    
    start = np.argmax(second < 0)
    
    return x.iloc[(np.argmax(second[start:] >= threshold)+start) or -1]


def ZPS_ee_grad_sub(x, expected_error_min, threshold=10, subsample=1, **kwargs):
    expected_error_min_sub = expected_error_min[1::subsample]
    grad = np.array(no_ahead_tvregdiff(expected_error_min_sub, 1, 1e-1, plotflag=False, diagflag=False))
    
    second = np.array(no_ahead_tvregdiff(grad[2:], 1, 15, plotflag=False, diagflag=False))
    
    start = np.argmax(second < 0)
    
    return x.iloc[1::subsample][2:][(np.argmax(second[start:] >= threshold)+start) or -1]


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


def SSNCut(config, classifiers, i, m=.2, affinity='linear', **kwargs):
    """
    NOTES:
    * They used an RBF svm to match the RBF affinity measure for SpectralClustering
    * As we carry out experiments on a linear svm we also use a linear affinity matrix (by default)
    
    file:///F:/Documents/Zotero/storage/DJGRDXSK/Fu%20and%20Yang%20-%202015%20-%20Low%20density%20separation%20as%20a%20stopping%20criterion%20for.pdf
    """
    unique_y = np.unique(classifiers[0].y_training)
    if len(unique_y) > 2:
        print("WARNING: SSNCut is not designed for non-binary classification")
        
    clustering = SpectralClustering(n_clusters=unique_y, affinity=affinity)
    
    out = []
    
    for clf in classifiers:
        X_unlabelled = reconstruct_unlabelled(config, clf, i)
        # Note: With non-binary classification the value of the decision function is a transformation of the distance...
        order = np.argsort(np.abs(clf.estimator.decision_function(X_unlabelled)))
        M = X_unlabelled[order[:min(1000, int(m*X_unlabelled.shape[0]))]]

        y0 = clf.predict(M)
        # use algorithm 1
        print("before fit_predict")
        print(M.shape, M.dtype, M.nnz, M.data.shape, M.indices.shape, M.indptr.shape)
        y1 = clustering.fit_predict(M)
        print("after fit_predict")

        diff = np.sum(y0==y1)/X_unlabelled.shape[0]
        if diff > 0.5:
            diff = 1 - diff
        out.append(diff)
            
    return out


def reconstruct_unlabelled(config, clf, i):
    import libdatasets
    import scipy
    X, y = getattr(libdatasets, config.dataset_name)(None)
    rand = check_random_state(i)
    if config.dataset_mutator_name != 'none':
        raise UnimplementedError("WARNING: Dataset mutation is not implemented in reconstruct_unlabelled")
    X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_teset, Y_test = active_split(
        X, y, labeled_size=config.meta['labelled_size'], test_size=config.meta['test_size'], random_state=rand, ensure_y=config.meta['ensure_y'])
    
    if not isinstance(X, scipy.sparse.csr_matrix):
        return X_unlabelled[(X_unlabelled[:,np.newaxis]!=clf.X_training).all(-1).any(-1)]
    
    def compare(A, B):
        "https://stackoverflow.com/questions/23124403/how-to-compare-2-sparse-matrix-stored-using-scikit-learn-library-load-svmlight-f"
        return zip(*np.where((np.array(A.multiply(A).sum(1)) +
            np.array(B.multiply(B).sum(1)).T) - 2 * A.dot(B.T).toarray() == 0))
    
    from libactive import delete_from_csr

    print(list(compare(X_unlabelled, clf.X_training)))
    return delete_from_csr(X_unlabelled, compare(X_unlabelled, clf.X_training))


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


def ZPS(classifiers, order=1, **kwargs):
    """
    Determine a stopping point based on the accuracy of previously trained classifiers.
    """
    accx, _acc = first_acc(classifiers)
    grad = np.array(no_ahead_tvregdiff(_acc, 1, 1e-1, plotflag=False, diagflag=False))
    start = np.argmax(grad < 0)
    
    if order == 2:
        second = np.array([np.nan, np.nan, *no_ahead_tvregdiff(grad[2:], 1, 1e-1, plotflag=False, diagflag=False)])
        return accx[np.argmax((grad[start:] >= 0) & (second[start:] >= 0)) + start]
    
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
            
        unique_labels = np.unique(clf.y_training)
        if metric == roc_auc_score:
            if len(unique_labels) > 2 or len(clf.y_training.shape) > 1:
                diffs.append(metric(clf.y_training[-size:], pclf.predict_proba(clf.X_training[-size:]), multi_class="ovr"))
            else:
                diffs.append(metric(clf.y_training[-size:], pclf.predict_proba(clf.X_training[-size:])[:,1]))
        elif metric == f1_score:
            diffs.append(metric(
                clf.y_training[-size:], 
                pclf.predict(clf.X_training[-size:]),
                average="micro" if len(unique_labels) > 2 else "binary",
                pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1
            ))
                    
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
        
        unique_labels = np.unique(clf.y_training)
        if metric == roc_auc_score:
            prediction = pclf.predict_proba(clf.X_training)
            try:
                if len(unique_labels) > 2 or len(clf.y_training.shape) > 1:
                    diffs.append(metric(clf.y_training, prediction, multi_class="ovr"))
                else:
                    diffs.append(metric(clf.y_training, prediction[:,1]))
            except ValueError:
                print(prediction)
        elif metric == f1_score:
            diffs.append(metric(
                clf.y_training[-size:], 
                pclf.predict(clf.X_training[-size:]),
                average="micro" if len(unique_labels) > 2 else "binary",
                pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1
            ))
        else:
            diffs.append(metric(clf.y_training, pclf.predict(clf.X_training)))
        
    return x, diffs


def kappa_metric(x, classifiers, k=3, **kwargs):
    from sklearn.preprocessing import OneHotEncoder
    
    out = [np.nan, np.nan]
    for i, clf in enumerate(classifiers[k-1:]):
        clfs = [*[classifiers[-i] for i in range(1, k)], clf]
        
        # FIXME: Shouldn't this be clf.predict(clfs[-1].X_training) ?
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
# Evaluation
# ----------------------------------------------------------------------------------------------

def eval_stopping_conditions(results_plots, classifiers, conditions=None):
    if conditions is None:
        params = {
            "kappa": {"k": 2}
        }
        conditions = {
            **{f"{f.__name__}": partial(f, **params.get(f.__name__, {})) for f in [uncertainty_min, SC_entropy_mcs, SC_oracle_acc_mcs, EVM, SC_mes, ZPS_ee_grad]},
            "ZPS2": partial(ZPS, order=2)
        }

    stop_results = {}
    for (clfs, (conf, metrics)) in zip(classifiers, results_plots):
        stop_results[conf.dataset_name] = {}
        for (name, cond) in conditions.items():
            stop_results[conf.dataset_name][name] = [cond(**metric, classifiers=clfs_, config=conf) for clfs_, metric in zip(clfs, metrics)]
            
    return (conditions, stop_results)


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
    
def rank_stop_conds(stop_results, results_plots, metric, title=None, holistic_x=50):
    data = []
    # n instances data
    for i, dataset in enumerate(stop_results.keys()):
        for ii, method in enumerate(stop_results[dataset].keys()):
            if i == 0:
                data.append([])
            for iii, run in enumerate(stop_results[dataset][method]):
                if metric == "instances":
                    data[ii].append(run)
                elif metric == "holistic":
                    data[ii].append(
                        (results_plots[i][1][iii].accuracy_score[results_plots[i][1][iii].x==run].iloc[0]+results_plots[i][1][iii].roc_auc_score[results_plots[i][1][iii].x==run].iloc[0])/2*holistic_x*100-run
                    )
                else:
                    data[ii].append(results_plots[i][1][iii][metric][results_plots[i][1][iii].x==run].iloc[0])
    data = pd.DataFrame(np.array(data).T, columns=list(stop_results[list(stop_results.keys())[0]].keys()))

    autoranked = autorank(data, order='ascending' if metric == 'instances' else 'descending')

    ax = plot_stats(autoranked)
    ax.figure.suptitle(title or metric.rsplit("_score")[0].replace("_", " ").title());
    return ax