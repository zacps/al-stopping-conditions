from typing import Tuple, Union, Callable
import time
from copy import deepcopy
import pickle
import zlib

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from celluloid import Camera
from IPython.core.display import HTML, display
from modAL import batch, density, disagreement, uncertainty, utils
from modAL.models import ActiveLearner, Committee
from sklearn import datasets, metrics, tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone
from sklearn.svm import SVC
from modAL.utils.data import data_vstack
from modAL.utils.selection import multi_argmax
from modAL.uncertainty import _proba_uncertainty, _proba_entropy, classifier_uncertainty
import scipy

from libadversarial import fgm, deepfool
from libplot import plot_classification, plot_poison, c_plot_poison
from libutil import Metrics

# Use GPU-based thundersvm when available
try:
    from thundersvm import SVC as ThunderSVC
except Exception:
    pass


def active_split(X, Y, test_size=0.5, labeled_size=0.1, shuffle=True, ensure_y=False, random_state=None, mutator=lambda *args: args, config_str=None, i=None):
    """
    Split data into three sets:
    * Labeled training set (0.1)
    * Unlabeled training set, to be queried (0.4)
    * Labeled test (0.5)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    X_train, X_test, Y_train, Y_test = mutator(X_train, X_test, Y_train, Y_test, rand=random_state, config_str=config_str, i=i)
    X_labelled, X_unlabelled, Y_labelled, Y_oracle = train_test_split(
        X_train,
        Y_train,
        test_size=(1 - labeled_size / test_size),
        shuffle=shuffle,
        random_state=random_state,
    )
    # ensure a label for all classes made it in to the initial train and validation sets
    if ensure_y:
        for klass in np.unique(Y):
            if klass not in Y_labelled:
                idx = np.where(Y_oracle==klass)[0][0]
                Y_labelled = np.concatenate((Y_labelled, [Y_oracle[idx]]), axis=0)
                X_labelled = np.concatenate((X_labelled, [X_unlabelled[idx]]), axis=0)
                Y_oracle = np.delete(Y_oracle, idx, axis=0)
                X_unlabelled = np.delete(X_unlabelled, idx, axis=0)

    return X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_test, Y_test


def active_split_query_synthesis(X, Y, test_size=0.5, labeled_size=0.1, shuffle=True, random_state=None):
    """
    Split data into three sets:
    * Labeled training set (0.1)
    * Unlabeled training set, to be queried (0.4)
    * Labeled test (0.5)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )
    X_labelled, X_unlabelled, Y_labelled, Y_oracle = train_test_split(
        X_train,
        Y_train,
        test_size=(1 - labeled_size / test_size),
        shuffle=shuffle,
        random_state=random_state,
    )

    return X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_test, Y_test


class MyActiveLearner:
    def __init__(
        self,
        animate=False,
        metrics=None,
        poison=False,
        animation_file=None,
        lb=None,
        ub=None,
    ):
        self.animate = animate
        self.metrics = Metrics(metrics=metrics)
        self.animation_file = animation_file
        self.poison = poison

        self.lb = lb
        self.ub = ub

        if self.animate:
            if poison:
                self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10))
            else:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
            self.cam = Camera(self.fig)

    def __setup_learner(self, X_labelled, Y_labelled, query_strategy, model):
        if model == "svm-linear":
            return ActiveLearner(
                estimator=SVC(kernel="linear", probability=True),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "thunder-svm-linear":
            return ActiveLearner(
                estimator=ThunderSVC(kernel="linear", probability=True),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "svm-rbf":
            return ActiveLearner(
                estimator=SVC(kernel="rbf", probability=True),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "svm-poly":
            return ActiveLearner(
                estimator=SVC(kernel="poly", probability=True),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "random-forest":
            return ActiveLearner(
                estimator=RandomForestClassifier(),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "gaussian-nb":
            return ActiveLearner(
                estimator=GaussianNB(),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "k-neighbors":
            return ActiveLearner(
                estimator=KNeighborsClassifier(),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "perceptron":
            return ActiveLearner(
                estimator=Perceptron(),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "committee":
            return Committee(
                learner_list=[
                    ActiveLearner(
                        estimator=SVC(kernel="linear", probability=True),
                        X_training=X_labelled,
                        y_training=Y_labelled,
                    ),
                    # committee: logistic regression, svm-linear, svm-rbf, guassian process classifier
                    ActiveLearner(
                        estimator=SVC(kernel="rbf", probability=True),
                        X_training=X_labelled,
                        y_training=Y_labelled,
                    ),
                    ActiveLearner(
                        estimator=GaussianProcessClassifier(),
                        X_training=X_labelled,
                        y_training=Y_labelled,
                    ),
                    ActiveLearner(
                        estimator=LogisticRegression(),
                        X_training=X_labelled,
                        y_training=Y_labelled,
                    ),
                ],
                query_strategy=disagreement.vote_entropy_sampling,
            )
        else:
            raise Exception("unknown model")

    def __animation_frame(
        self,
        learner,
        X_unlabelled=None,
        new=None,
        new_labels=None,
        start_points=None,
        ax=None,
    ):
        if ax is None:
            ax = self.ax
        if X_unlabelled is not None:
            ax.scatter(X_unlabelled[:, 0], X_unlabelled[:, 1], c="black", s=20)

        plot_classification(
            ax,
            learner.estimator,
            learner.X_training,
            learner.y_training,
            np.concatenate((learner.X_training, X_unlabelled), axis=0)
            if X_unlabelled
            else learner.X_training,
        )

        ax.text(
            0.9,
            0.05,
            str(
                learner.X_training.shape[0],
            ),
            transform=ax.transAxes,
            c="white",
        )
        self.cam.snap()

    def __animation_frame_poison(
        self, learner, X_test, y_test, attack_points, start_points, ax=None
    ):
        if ax is None:
            ax = self.ax
        plot_poison(
            clf=learner.estimator,
            X_labelled=learner.X_training,
            y_labelled=learner.y_training,
            X_unlabelled=None,
            y_unlabelled=None,
            X_test=X_test,
            y_test=y_test,
            attack=None,
            attack_points=attack_points,
            start_points=start_points,
            start_points_y=learner.estimator.predict(start_points),
            ax=ax,
        )

        ax.text(
            0.9,
            0.05,
            str(
                learner.X_training.shape[0],
            ),
            transform=ax.transAxes,
            c="white",
        )

    def __animation_frame_poison_c(
        self, learner, attack, lb, ub, attack_points, start_points, x_seq=None, ax=None
    ):
        if ax is None:
            ax = self.ax
        c_plot_poison(
            X_labelled=learner.X_training,
            y_labelled=learner.y_training,
            attack=attack,
            lb=lb,
            ub=ub,
            attack_points=attack_points,
            start_points=start_points,
            start_points_y=learner.estimator.predict(start_points)
            if start_points is not None
            else None,
            x_seq=x_seq,
            ax=ax,
        )

        ax.text(
            0.9,
            0.05,
            str(
                learner.X_training.shape[0],
            ),
            transform=ax.transAxes,
            c="white",
        )

    def active_learn2(
        self,
        X_labelled,
        X_unlabelled,
        Y_labelled,
        Y_oracle,
        X_test,
        Y_test,
        query_strategy,
        model="svm-linear",
        teach_advesarial=False,
        stop_function=lambda learner: False,
        ret_classifiers=False,
        stop_info=False,
        compress=False,
        config_str=None,
        i=None
    ) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset, querying data with the given query strategy.

        Returns metrics describing the performance of the query strategy, and optionally all classifiers trained during learning.
        """
        
        cached = _restore_run(config_str, i)
        if cached is not None:
            return cached

        checkpoint = _restore_checkpoint(config_str, i)
        if checkpoint is None:
            learner = self.__setup_learner(
                X_labelled, Y_labelled, query_strategy, model=model
            )
            classifiers = []
            if ret_classifiers:
                classifiers.append(deepcopy(learner))
            self.metrics.collect(X_labelled.shape[0], learner.estimator, Y_test, X_test)
        else:
            learner, self.metrics, X_unlabelled, Y_oracle = checkpoint

        while X_unlabelled.shape[0] != 0 and not stop_function(learner):
            t_start = time.monotonic()
            if not stop_info:
                query_idx, query_points = learner.query(X_unlabelled)
                extra_metrics = {}
            else:
                query_idx, query_points, extra_metrics = learner.query(X_unlabelled)
            t_elapsed = time.monotonic() - t_start
            
            if "expected_error" in self.metrics.metrics:
                extra_metrics['expected_error'] = expected_error(learner, X_unlabelled)
            if "contradictory_information" in self.metrics.metrics:
                # https://stackoverflow.com/questions/32074239/sklearn-getting-distance-of-each-point-from-decision-boundary
                predictions = learner.predict(query_points)
                uncertainties = classifier_uncertainty(learner.estimator, query_points)

            if query_points is not None and getattr(query_strategy, "is_adversarial", False):
                learner.teach(query_points, Y_oracle[query_idx])

            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])
            
            if "contradictory_information" in self.metrics.metrics:
                contradictory_information = np.sum(uncertainties[predictions != Y_oracle[query_idx]]/np.mean(uncertainties))
                extra_metrics['contradictory_information'] = contradictory_information

            if isinstance(X_unlabelled, csr_matrix):
                X_unlabelled = delete_from_csr(X_unlabelled, row_indices=query_idx)
            else:
                X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
                
            Y_oracle = np.delete(Y_oracle, query_idx, axis=0)

            self.metrics.collect(
                self.metrics.frame.x.iloc[-1] + len(query_idx),
                learner.estimator,
                Y_test,
                X_test,
                time=t_elapsed,
                X_unlabelled=X_unlabelled,
                **extra_metrics
            )
                
            if ret_classifiers:
                classifiers.append(deepcopy(learner))
                
            _checkpoint(config_str, i, (learner, self.metrics, X_unlabelled, Y_oracle))
                
        if not ret_classifiers:
            _write_run(config_str, i, self.metrics)
            _cleanup_checkpoint(config_str, i)
            return (self.metrics, None)
        else:
            _write_run(config_str, i, self.metrics, classifiers)
            _cleanup_checkpoint(config_str, i)
            return (self.metrics, zlib.compress(pickle.dumps(classifiers)) if compress else classifiers)

    def active_learn_query_synthesis(
        self,
        X_labelled,
        Y_labelled,
        y_oracle: Callable,
        X_test,
        Y_test,
        query_strategy,
        should_stop: Callable,
        model="svm-linear",
        teach_advesarial=False,
        track_flips=False,
    ) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

        Returns metrics.
        """

        total_labels = 0
        oracle_matched_poison = 0

        learner = self.__setup_learner(
            X_labelled, Y_labelled, query_strategy, model="svm-linear"
        )

        self.metrics.collect(len(X_labelled), learner.estimator, Y_test, X_test)

        if self.animate and not self.poison:
            self.__animation_frame(learner)
        elif self.animate and self.poison:
            self.__animation_frame(learner, ax=self.ax[0])
            self.__animation_frame_poison_c(
                learner,
                None,
                lb=self.lb,
                ub=self.ub,
                attack_points=None,
                start_points=None,
                ax=self.ax[1],
            )

        while not should_stop(learner, self.metrics.frame.iloc[-1]):
            try:
                t_start = time.monotonic()
                (
                    _,
                    query_points,
                    start_points,
                    attack,
                    x_seq,
                    query_points_labels,
                ) = learner.query(None, learner.X_training, learner.y_training)
                t_elapsed = time.monotonic() - t_start
            except np.linalg.LinAlgError:
                print("WARN: Break due to convergence failure")
                break

            labels = y_oracle(query_points)
            total_labels += len(labels)
            oracle_matched_poison += np.count_nonzero(labels == query_points_labels)

            learner.teach(query_points, labels)

            self.metrics.collect(
                self.metrics.frame.x.iloc[-1] + len(query_points),
                learner.estimator,
                Y_test,
                X_test,
                t_elapsed=t_elapsed,
            )

            if self.animate and not self.poison:
                self.__animation_frame(
                    learner,
                    new=query_points,
                    new_labels=labels,
                    start_points=start_points,
                )
            elif self.animate and self.poison:
                self.__animation_frame_poison(
                    learner, X_test, Y_test, query_points, start_points, ax=self.ax[0]
                )
                self.__animation_frame_poison_c(
                    learner,
                    attack,
                    lb=self.lb,
                    ub=self.ub,
                    attack_points=query_points,
                    start_points=start_points,
                    x_seq=x_seq,
                    ax=self.ax[1],
                )
                self.cam.snap()

        if self.animate:
            animation = self.cam.animate(interval=500, repeat_delay=1000)
            if self.animation_file is not None:
                animation.save(animation_file)
            display(HTML(animation.to_html5_video()))
            plt.close(self.fig)

        if track_flips:
            print(
                f"{oracle_matched_poison} of {total_labels} had equal oracle and poison attack labels"
            )

        return self.metrics

    
def _checkpoint(config_str, i, data):
    file = f"cache/checkpoints/{config_str}_{i}.pickle"
    with open(file, "wb") as f:
        pickle.dump(data, f)
    

def _restore_checkpoint(config_str, i):
    file = f"cache/checkpoints/{config_str}_{i}.pickle"
    try:
        with open(file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None
    
    
def _cleanup_checkpoint(config_str, i):
    file = f"cache/checkpoints/{config_str}_{i}.pickle"
    try:
        os.remove(file)
    except FileNotFoundError:
        pass


def _write_run(config_str, i, metrics):
    file = f"cache/runs/{config_str}_{i}.csv"
    with open(file, "w") as f:
        metrics.frame.to_csv(f)


def _restore_run(config_str, i):
    file = f"cache/runs/{config_str}_{i}.csv"
    try:
        with open(file, "r") as f:
            metrics = pd.read_csv(f, index_col=0)
    except FileNotFoundError:
        return None
    return metrics
        
    
class BeamClf:
    def __init__(self, X_labelled, y_labelled, X_unlabelled, y_unlabelled, X_test, y_test):
        self._ = SVC(kernel='linear', probability=True)
        self._.fit(X_labelled, y_labelled)
        self.X = X_labelled
        self.y = y_labelled
        self.X_unlabelled = X_unlabelled
        self.y_unlabelled = y_unlabelled
        self.X_test = X_test
        self.y_test = y_test
        self.metrics = Metrics()
        self.metrics.collect(len(self.X), self._, self.y_test, self.X_test)
        
    def teach(self, idx):
        self.X = np.concatenate((self.X, [self.X_unlabelled[idx]]), axis=0)
        self.y = np.concatenate((self.y, [self.y_unlabelled[idx]]), axis=0)
        self.X_unlabelled = np.delete(self.X_unlabelled, idx, axis=0)
        self.y_unlabelled = np.delete(self.y_unlabelled, idx, axis=0)
        
        self._.fit(self.X, self.y)
        self.metrics.collect(len(self.X), self._, self.y_test, self.X_test)
        
    def done(self):
        return len(self.X_unlabelled) == 0

def beam_search2(
    X_labelled,
    X_unlabelled,
    y_labelled,
    y_unlabelled,
    X_test,
    y_test,
    workers: int = 1,
    beam_width: int = 5,
    metric: str = "accuracy_score"
):
    """
    Perform beam-search on the split dataset. 
    
    This should generate a best-guess at the optimal active learning sequence for a dataset.
    """
    classifiers = [BeamClf(X_labelled, y_labelled, X_unlabelled, y_unlabelled, X_test, y_test)]
    
    while any(not clf.done() for clf in classifiers):
        temp_clfs = []
        for clf in classifiers:
            temp_clfs.append([])
            if workers == 1:
                for idx in range(len(clf.X_unlabelled)):
                    new = deepcopy(clf)
                    new.teach(idx)
                    temp_clfs[-1].append(new)
            else:
                def func(idx):
                    new = deepcopy(clf)
                    new.teach(idx)
                    return new
                temp_clfs[-1].extend(Parallel(n_jobs=workers)(delayed(lambda i: func(i))(idx) for idx in range(len(clf.X_unlabelled))))
                
        for l in temp_clfs:
            l.sort(key=lambda clf: clf.metrics.frame[metric].iloc[-1], reverse=True)
        
        classifiers = [clf for clf in l for l in temp_clfs][:beam_width]
        temp_clfs = []
    
    return classifiers[0]


def delete_from_csr(mat, row_indices=None, col_indices=None):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices is not None:
        rows = list(row_indices)
    if col_indices is not None:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat

    
def interactive_img_oracle(images):
    fig, axes = plt.subplots(len(images))
    for i, (ax, image) in enumerate(zip(np.array(axes).flatten(), images)):
        ax.imshow(image.reshape(8,8), cmap='gray',interpolation='none')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(str(i))
    display(fig)
    classes = []
    for i in range(len(images)):
        klass = input(f"Image {i} class?")
        if klass == "?":
            klass = 11
        classes.append(int(klass))
    return np.array(classes)

def expected_error(learner, X, predict_proba=None, p_subsample=1.0):
    loss = 'binary'
    
    expected_error = np.zeros(shape=(X.shape[0], ))
    possible_labels = np.unique(learner.y_training)

    try:
        X_proba = predict_proba or learner.predict_proba(X)
    except NotFittedError:
        # TODO: implement a proper cold-start
        return 0, X[0]

    cloned_estimator = clone(learner.estimator)

    for x_idx in range(X.shape[0]):
        # subsample the data if needed
        if np.random.rand() <= p_subsample:
            if isinstance(X, csr_matrix):
                X_reduced = delete_from_csr(X, [x_idx])
            else:
                X_reduced = np.delete(X, x_idx, axis=0)
            # estimate the expected error
            for y_idx, y in enumerate(possible_labels):
                if isinstance(X, csr_matrix):
                    X_new = scipy.sparse.vstack((learner.X_training, X[[x_idx]]))
                else:
                    X_new = data_vstack((learner.X_training, np.expand_dims(X[x_idx], axis=0)))
                y_new = data_vstack((learner.y_training, np.array(y).reshape(1,)))

                cloned_estimator.fit(X_new, y_new)
                refitted_proba = cloned_estimator.predict_proba(X_reduced)
                if loss == 'binary':
                    nloss = _proba_uncertainty(refitted_proba)
                elif loss == 'log':
                    nloss = _proba_entropy(refitted_proba)
                else:
                    raise Exception("unknown loss")

                expected_error[x_idx] += np.sum(nloss)*X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] = np.inf

    return expected_error