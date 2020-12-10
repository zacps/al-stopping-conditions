import sklearn
import numpy as np
from modAL import batch, uncertainty, density, utils, disagreement
from art.estimators.classification.scikitlearn import SklearnClassifier
from art.attacks.evasion import FastGradientMethod, DeepFool
from art.attacks.poisoning import PoisoningAttackSVM
from sklearn.metrics.pairwise import paired_distances
from sklearn import preprocessing
from typing import Union, Tuple, Callable
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

def random_batch(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    **uncertainty_measure_kwargs
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


def fgm(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    **uncertainty_measure_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    classifier = SklearnClassifier(model=classifier.estimator)
    attack = FastGradientMethod(estimator=classifier, eps=0.2)

    adversarial_examples = attack.generate(X)

    # TODO: I have no idea if this is the right way to rank them, it seems to be right intuitively but...
    dists = paired_distances(X, adversarial_examples, metric="euclidean")

    idx = np.argsort(dists)

    # This is kind of a hack, modAL is not built to deal with passing extra information.
    result = (idx[:n_instances], adversarial_examples[idx[:n_instances]])

    return result


def deepfool(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    n_instances: int = 1,
    random_tie_break: bool = False,
    verbose: bool = False,
    clip_values: Tuple[int, int] = None,
    **uncertainty_measure_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    classifier = SklearnClassifier(model=classifier.estimator, clip_values=clip_values)
    attack = DeepFool(classifier=classifier, verbose=verbose)

    adversarial_examples = attack.generate(X)

    # TODO: I have no idea if this is the right way to rank them, it seems to be right intuitively but...
    dists = paired_distances(X, adversarial_examples, metric="euclidean")

    idx = np.argsort(dists)

    # This is kind of a hack, modAL is not built to deal with passing extra information.
    result = (idx[:n_instances], adversarial_examples[idx[:n_instances]])

    return result


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
    Note this is a query-synthesis active learning method. That is it ignores the unlabelled set X.
    """
    
    X_train, X_validate, y_train, y_validate = train_test_split(X_training, y_training, test_size=0.5)
    
    classifier = SklearnClassifier(model=classifier.estimator, clip_values=clip_values)

    # Select start points for the poisoning attack
    start_points_idx, start_points = point_selector(classifier, X_training, n_instances=n_instances)
    y = classifier.predict(start_points)

    # switch binary classifier labels to multi label classifier labels
    y_training = np.array([y_training, np.logical_not(y_training)]).T
        
    attack = PoisoningAttackSVM(
        classifier, 
        x_train=X_train, 
        y_train=np.array([y_train, np.logical_not(y_train)]).T, 
        x_val=X_validate, 
        y_val=y_validate, 
        eps=1, # Original paper doesn't discuss values of this parameter at all...
        step=0.1, # nor this one...
        verbose=False
    )
    if invert:
        y = np.logical_not(y)
    points, _labels = attack.poison(start_points, y=y)
        
    return (None, points, start_points)

def random_synthesis(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    bounds: Tuple[Tuple[int, int], ...],
    *args,
    n_instances: int = 1,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    points = np.array([rng.uniform(low, high, n_instances) for (low, high) in bounds])
    return (None, points.T, None)

def uncertainty_synthesis(
    classifier: sklearn.base.BaseEstimator,
    X: Union[list, np.ndarray],
    X_training: Union[list, np.ndarray],
    y_training: Union[list, np.ndarray],
    bounds: Tuple[Tuple[int, int], ...],
    *args,
    n_instances: int = 1,
    **kwargs
):
    # generate a meshgrid then make confidence predictions for those points?
    # naive but should do as an approximation of the confidence measure
    # ... but only in this restricted low-dimension case I think. otherwise
    # the density of the grid will fall appart...
    pass