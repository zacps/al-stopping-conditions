import sklearn
import numpy as np
from modAL import batch, uncertainty, density, utils, disagreement
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool
from art.attacks.poisoning import PoisoningAttackSVM
from sklearn.metrics.pairwise import paired_distances, euclidean_distances
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from typing import Union, Tuple, Callable
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from secml.ml.classifiers.sklearn.c_classifier_svm import CClassifierSVM
from secml.adv.attacks.poisoning import CAttackPoisoningSVM
from secml.ml.kernels.c_kernel_linear import CKernelLinear
from secml.data.c_dataset import CDataset
from secml.data.data_utils import label_binarize_onehot
from secml.array.c_array import CArray


def random_batch(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    **uncertainty_measure_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    try:
        idx = rng.choice(X.shape[0], n_instances, replace=False)
    except ValueError:
        idx = rng.choice(X.shape[0], X.shape[0], replace=False)
    return (idx, X[idx])


def uncertainty_id(clf, X, n_instances=1, **kwargs):
    """
    Sort by the minimum highest confidence labelling.
    """
    return np.argsort(
        uncertainty.classifier_uncertainty(clf, X) * density.information_density(X)
    )[:n_instances]


# deprecated
def fgm(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    teach_adversarial: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier = SklearnClassifier(model=classifier.estimator)
    attack = FastGradientMethod(estimator=classifier, eps=0.2, **kwargs)

    adversarial_examples = attack.generate(X)

    # TODO: I have no idea if this is the right way to rank them, it seems to be right intuitively but...
    dists = paired_distances(X, adversarial_examples, metric="euclidean")

    idx = np.argsort(dists)

    # This is kind of a hack, modAL is not built to deal with passing extra information.
    result = (
        idx[:n_instances],
        adversarial_examples[idx[:n_instances]] if teach_adversarial else None,
    )

    return result


# deprecated
def deepfool(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    verbose: bool = False,
    clip_values: Tuple[int, int] = None,
    batch_size: int = 1,
    teach_adversarial: bool = False,
    **uncertainty_measure_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    classifier = SklearnClassifier(model=classifier.estimator, clip_values=clip_values)
    attack = DeepFool(classifier=classifier, verbose=verbose, batch_size=batch_size)

    adversarial_examples = attack.generate(X)

    # TODO: I have no idea if this is the right way to rank them, it seems to be right intuitively but...
    dists = paired_distances(X, adversarial_examples, metric="euclidean")

    idx = np.argsort(dists)

    # This is kind of a hack, modAL is not built to deal with passing extra information.
    result = (
        idx[:n_instances],
        adversarial_examples[idx[:n_instances]] if teach_adversarial else None,
    )

    return result


def adversarial(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    Attack,
    n_instances: int = 1,
    clip_values: Tuple[int, int] = None,
    teach_adversarial: bool = False,
    parallel_threads: int = 1,
    use_logits: bool = False,
    **attack_kwargs,
):
    """
    Generic adversarial margin-based pool active learning query strategy.

    Optionally runs attacks in parallel.
    """

    classifier = SklearnClassifier(
        model=classifier.estimator, clip_values=clip_values, use_logits=use_logits
    )

    attack = Attack(classifier, **attack_kwargs)

    if parallel_threads != 1:
        # *Should* be ordered, at least with the default backend
        # multiprocessing backend may not be ordered
        adversarial_examples = Parallel(n_jobs=parallel_threads)(
            delayed(attack.generate)(part)
            for part in np.array_split(x, parallel_threads)
        )
        adversarial_examples = np.array(adversarial_examples).reshape(-1, X.shape[-1])
    else:
        adversarial_examples = attack.generate(X)

    dists = paired_distances(X, adversarial_examples, metric="euclidean")

    idx = np.argsort(dists)

    # This is kind of a hack, modAL is not built to deal with passing extra information.
    result = (
        idx[:n_instances],
        adversarial_examples[idx[:n_instances]] if teach_adversarial else None,
    )

    return result


def density(
    classifier: sklearn.base.BaseEstimator,
    X_unlabelled: Union[list, np.ndarray],
    n_instances: int = 1,
    **attack_kwargs,
):
    """
    Low density pool-based strategy.

    Looks for low density reigons in the current labelled set.
    """

    X_labelled = classifier.X_training

    # Calculate similarity of each unlabelled point to our current labelled set
    # by calculating pairwise distances and summing accross the unlabelled points.
    density = np.sum(euclidean_distances(X_unlabelled, X_labelled), axis=1)

    idx = np.argsort(-density)

    return (idx[:n_instances], None)


def comparative_density(
    classifier: sklearn.base.BaseEstimator,
    X_unlabelled: Union[list, np.ndarray],
    n_instances: int = 1,
    **attack_kwargs,
):
    """
    Low density pool-based strategy.

    Looks for low density reigons in the current labelled set.
    """

    X_labelled = classifier.X_training

    # Calculate similarity of each unlabelled point to our current labelled set
    # by calculating pairwise distances and summing accross the unlabelled points,
    # then subtracting the distance from the other unlabelled points.
    #
    # This is to try to avoid only selecting outliers.
    density = np.sum(euclidean_distances(X_unlabelled, X_labelled), axis=1) - np.sum(
        euclidean_distances(X_unlabelled, X_unlabelled), axis=1
    )

    idx = np.argsort(-density)

    return (idx[:n_instances], None)


def comparative_density_margin(
    classifier: sklearn.base.BaseEstimator,
    X_unlabelled: Union[list, np.ndarray],
    n_instances: int = 1,
    clip_values: Tuple[int, int] = None,
    **attack_kwargs,
):
    """
    Low density pool-based strategy.

    Looks for comparatively low density reigons in the current labelled set. Also takes into account distance to margin.
    """

    X_labelled = classifier.X_training

    # Calculate similarity of each unlabelled point to our current labelled set
    # by calculating pairwise distances and summing accross the unlabelled points,
    # then subtracting the distance from the other unlabelled points.
    #
    # This is to try to avoid only selecting outliers.
    density = np.sum(euclidean_distances(X_unlabelled, X_labelled), axis=1) - np.sum(
        euclidean_distances(X_unlabelled, X_unlabelled), axis=1
    )
    # Normalize
    density = density / density.max()

    # Margin finding
    classifier = SklearnClassifier(model=classifier.estimator, clip_values=clip_values)
    attack = FastGradientMethod(classifier, minimal=True)

    adversarial_examples = attack.generate(X_unlabelled)

    dists = paired_distances(X_unlabelled, adversarial_examples, metric="euclidean")
    # Normalize
    dists = dists / dists.max()

    # High density distance = good, high margin distance = bad
    score = dists - density

    idx = np.argsort(score)

    return (idx[:n_instances], None)


def poison(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    *args,
    n_instances: int = 1,
    point_selector: Callable = random_batch,
    clip_values: Tuple[int, int] = None,
    invert: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note this is a query-synthesis active learning method.
    """

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_training, y_training, test_size=0.5
    )

    classifier = SklearnClassifier(model=classifier.estimator, clip_values=clip_values)

    # Select start points for the poisoning attack
    start_points_idx, start_points = point_selector(
        classifier, X_training, n_instances=n_instances
    )
    y = classifier.predict(start_points)

    # switch binary classifier labels to multi label classifier labels
    y_training = np.array([y_training, np.logical_not(y_training)]).T

    attack = PoisoningAttackSVM(
        classifier,
        x_train=X_train,
        y_train=np.array([y_train, np.logical_not(y_train)]).T,
        x_val=X_validate,
        y_val=y_validate,
        eps=2,  # Original paper doesn't discuss values of this parameter at all...
        step=1e-4,  # nor this one...
        verbose=True,
    )
    if invert:
        y = np.logical_not(y)
    points, _labels = attack.poison(start_points, y=y)

    return (None, points, start_points, None, None)


def poison_secml(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    *args,
    n_instances: int = 1,
    point_selector: Callable = random_batch,
    clip_values: Tuple[int, int] = None,
    invert: bool = False,
    lb=None,
    ub=None,
    solver_params=None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Note this is a query-synthesis active learning method.
    """
    if solver_params is None:
        solver_params = {
            "eta": 0.05,
            "eta_min": 0.05,
            "eta_max": None,
            "max_iter": 100,
            "eps": 1e-6,
        }

    y_training = LabelEncoder().fit_transform(y_training)

    X_train, X_validate, y_train, y_validate = train_test_split(
        X_training, y_training, test_size=0.5, shuffle=False
    )

    secclf = CClassifierSVM(kernel=CKernelLinear())
    # print("y_train", y_train)
    secclf.fit(CArray(X_train), CArray(y_train))

    # Select start points for the poisoning attack
    start_points_idx, start_points = point_selector(
        classifier, X_train, n_instances=n_instances
    )
    start_points_labels = y_train[start_points_idx]

    attack = CAttackPoisoningSVM(
        secclf,
        CDataset(X_train, y_train),
        CDataset(X_validate, y_validate),
        solver_params=solver_params,
        lb=CArray(lb),
        ub=CArray(ub),
    )
    attack.n_points = n_instances

    labels, _score, points, _final_objective = attack.run(
        start_points, CArray(start_points_labels)
    )
    points = points.X.tondarray()

    return (None, points, start_points, attack, attack.x_seq, labels)


def random_synthesis(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    bounds: Tuple[Tuple[int, int], ...],
    *args,
    n_instances: int = 1,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    points = np.array([rng.uniform(low, high, n_instances) for (low, high) in bounds])
    return (None, points.T, None, None, None, None)


def meshgrid_synthesis(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    bounds: Tuple[Tuple[int, int], ...],
    pool_strategy: Callable,
    *args,
    n_instances: int = 1,
    n_points: int = 200,
    **kwargs,
):
    dimensions = np.meshgrid(
        *[
            np.arange(
                bounds[i][0],
                bounds[i][1],
                (bounds[i][1] - bounds[i][0])
                / np.power(n_points, 1 / X_training.shape[-1]),
            )
            for i in range(X_training.shape[-1])
        ]
    )
    grid = np.column_stack(tuple(dimension.ravel() for dimension in dimensions))

    # in theory uncertainty batch should return points and indicies, but it seems to only return indicies for some reason...
    points_idx = pool_strategy(classifier, grid, n_instances=n_instances)
    return (points_idx, grid[points_idx], None, None, None, None)


def uncertainty_synthesis(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    bounds: Tuple[Tuple[int, int], ...],
    *args,
    n_instances: int = 1,
    **kwargs,
):
    import scipy
    import modAL

    def func(x, clf):
        return -modAL.uncertainty.classifier_uncertainty(clf, [x])

    out = []
    for _ in range(n_instances):
        p = np.random.uniform(
            np.array(bounds)[:, 0], np.array(bounds)[:, 1], X_training.shape[-1]
        )

        # TODO: Try basin-hopping global optimization
        # Possibly too computationally intensive
        out.append(
            scipy.optimize.minimize(func, args=classifier, x0=tuple(p), bounds=bounds).x
        )

    return (None, out, None, None, None, None)
