from typing import Tuple, Callable
import time
from copy import deepcopy
import pickle
import os
import zipfile
from contextlib import contextmanager

import sklearn
import dill
import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera

try:
    from IPython.core.display import display
except ModuleNotFoundError:
    pass
from modAL import disagreement
from modAL.models import ActiveLearner, Committee
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from sklearn import calibration
from sklearn.svm import SVC
from modAL.utils.data import data_vstack
from modAL.uncertainty import _proba_uncertainty, classifier_uncertainty
import scipy

from libplot import plot_classification, plot_poison, c_plot_poison
from libutil import Metrics, out_dir
from libstore import store


def active_split(
    X,
    Y,
    test_size=0.5,
    labeled_size=0.1,
    shuffle=True,
    ensure_y=False,
    random_state=None,
    mutator=lambda *args, **kwargs: args,
    config_str=None,
    i=None,
):
    """
    Split data into three sets:
    * Labeled training set (0.1)
    * Unlabeled training set, to be queried (0.4)
    * Labeled test (0.5)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=shuffle, random_state=random_state
    )

    # Apply a mutator (noise, unbalance, bias, etc) to the dataset
    X_unlabelled, X_test, Y_oracle, Y_test = mutator(
        X_train,
        X_test,
        Y_train,
        Y_test,
        rand=random_state,
        config_str=config_str,
        i=i,
        test_size=test_size,
        shuffle=shuffle,
    )

    unique = np.unique(np.concatenate((Y_test, Y_oracle)))

    X_labelled = (
        np.empty((0, X_unlabelled.shape[1]))
        if not isinstance(X, scipy.sparse.csr_matrix)
        else scipy.sparse.csr_matrix((0, X_unlabelled.shape[1]))
    )
    Y_labelled = np.empty(0 if len(Y_oracle.shape) == 1 else (0, Y_oracle.shape[1]))

    # ensure a label for all classes made it in to the initial train and validation sets
    for klass in unique:
        if not np.isin(klass, Y_labelled):
            # First value chosen is effectively constant random as the dataset is shuffled
            idx = np.where(Y_oracle == klass)[0][0]

            Y_labelled = np.concatenate((Y_labelled, [Y_oracle[idx]]), axis=0)

            if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
                X_labelled = csr_vappend(X_labelled, X_unlabelled[idx])
            else:
                X_labelled = np.concatenate((X_labelled, [X_unlabelled[idx]]), axis=0)
            Y_oracle = np.delete(Y_oracle, idx, axis=0)
            if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
                X_unlabelled = delete_from_csr(X_unlabelled, row_indices=[idx])
            else:
                X_unlabelled = np.delete(X_unlabelled, idx, axis=0)

    if labeled_size < 1:
        labeled_size = labeled_size * X.shape[0]

    if X_labelled.shape[0] < labeled_size:
        idx = random_state.choice(
            X_unlabelled.shape[0], labeled_size - X_labelled.shape[0], replace=False
        )

        Y_labelled = np.concatenate((Y_labelled, Y_oracle[idx]), axis=0)

        if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
            X_labelled = csr_vappend(X_labelled, X_unlabelled[idx])
        else:
            X_labelled = np.concatenate((X_labelled, X_unlabelled[idx]), axis=0)
        Y_oracle = np.delete(Y_oracle, idx, axis=0)
        if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
            X_unlabelled = delete_from_csr(X_unlabelled, row_indices=idx)
        else:
            X_unlabelled = np.delete(X_unlabelled, idx, axis=0)

    # Sanity checks
    assert (
        X_labelled.shape[0] == Y_labelled.shape[0]
        and Y_labelled.shape[0] >= labeled_size
    ), "Labelled length inconsistent"
    assert X_unlabelled.shape[0] == Y_oracle.shape[0], "Unlabelled length inconsistent"
    assert X_test.shape[0] == Y_test.shape[0], "Test length inconsistent"
    assert (
        X_labelled.shape[1] == X_unlabelled.shape[1] == X_test.shape[1]
    ), "X shape inconsistent"

    return X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_test, Y_test


class MyActiveLearner:
    def __init__(
        self,
        X_labelled,
        X_unlabelled,
        Y_labelled,
        Y_oracle,
        X_test,
        Y_test,
        query_strategy,
        config,
        metrics=None,
        i=None,
    ):
        self.X_labelled = X_labelled
        self.X_unlabelled = X_unlabelled
        self.Y_labelled = Y_labelled
        self.Y_oracle = Y_oracle
        self.X_test = X_test
        self.Y_test = Y_test
        self.query_strategy = query_strategy

        self.unique_labels = np.unique(Y_test)

        self.stop_function = config.meta.get(
            "stop_function", ("default", lambda learner: False)
        )[1]
        self.ret_classifiers = config.meta.get("ret_classifiers", False)
        self.stop_info = config.meta.get("stop_info", False)
        self.config_str = config.serialize()
        self.i = i
        self.pool_subsample = config.meta.get("pool_subsample", None)
        self.model = config.model_name.lower()

        self.metrics = Metrics(metrics=metrics)

        # Validate expected error method
        ee = config.meta.get("ee", "offline")
        if ee == "online":
            self.ee = expected_error_online
        elif ee == "offline":
            self.ee = expected_error
        else:
            raise ValueError(f"ee must be online or offline, got {ee}")

    def __setup_learner(self):
        if self.model == "svm-linear":
            return ActiveLearner(
                estimator=SVC(kernel="linear", probability=True),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "thunder-svm-linear":
            return ActiveLearner(
                estimator=ThunderSVC(kernel="linear", probability=True),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "svm-rbf":
            return ActiveLearner(
                estimator=SVC(kernel="rbf", probability=True),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "svm-poly":
            return ActiveLearner(
                estimator=SVC(kernel="poly", probability=True),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "random-forest":
            return ActiveLearner(
                estimator=RandomForestClassifier(),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "decision-tree":
            return ActiveLearner(
                estimator=DecisionTreeClassifier(),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "gaussian-nb":
            return ActiveLearner(
                estimator=GaussianNB(),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "k-neighbors":
            return ActiveLearner(
                estimator=KNeighborsClassifier(),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "perceptron":
            return ActiveLearner(
                estimator=Perceptron(),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "neural-network":
            return ActiveLearner(
                estimator=MLPClassifier(
                    hidden_layer_sizes=(100,),  # default
                    activation="relu",  # default
                ),
                X_training=self.X_labelled,
                y_training=self.Y_labelled,
                query_strategy=self.query_strategy,
            )
        elif self.model == "committee":
            return Committee(
                learner_list=[
                    ActiveLearner(
                        estimator=SVC(kernel="linear", probability=True),
                        X_training=self.X_labelled,
                        y_training=self.Y_labelled,
                    ),
                    # committee: logistic regression, svm-linear, svm-rbf, guassian process classifier
                    ActiveLearner(
                        estimator=SVC(kernel="rbf", probability=True),
                        X_training=self.X_labelled,
                        y_training=self.Y_labelled,
                    ),
                    ActiveLearner(
                        estimator=GaussianProcessClassifier(),
                        X_training=self.X_labelled,
                        y_training=self.Y_labelled,
                    ),
                    ActiveLearner(
                        estimator=LogisticRegression(),
                        X_training=self.X_labelled,
                        y_training=self.Y_labelled,
                    ),
                ],
                query_strategy=disagreement.vote_entropy_sampling,
            )
        else:
            raise Exception("unknown model")

    def active_learn2(self) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset, querying data with the given query strategy.

        Returns metrics describing the performance of the query strategy, and optionally all classifiers trained during learning.
        """

        # If this experiment run has been completed previously return the saved result
        cached = self._restore_run()
        if cached is not None:
            return cached

        # Attempt to restore a checkpoint
        checkpoint = self._restore_checkpoint()

        if checkpoint is None:
            self.learner = self.__setup_learner()
            self.metrics.collect(
                self.X_labelled.shape[0],
                self.learner.estimator,
                self.Y_test,
                self.X_test,
            )
        else:
            print("Restoring from checkpoint")
            self = checkpoint

        # Classifiers are stored as a local and explicitly restored as they need to be compressed before being stored.
        with store(
            f"{out_dir()}/classifiers/{self.config_str}_{self.i}.zip",
            enable=self.ret_classifiers,
            restore=checkpoint is not None,
        ) as classifiers:
            # Initial subsampling, this should probably be done somewhere else tbh...
            if checkpoint is None:
                if self.pool_subsample is not None:
                    # TODO: Should this random be seeded?
                    self.X_subsampled = self.X_unlabelled[
                        np.random.choice(
                            self.X_unlabelled.shape[0],
                            self.pool_subsample,
                            replace=False,
                        )
                    ]
                else:
                    self.X_subsampled = self.X_unlabelled

            if self.ret_classifiers and len(classifiers) == 0:
                # Store the unlabelled pool with the classifier
                # self.learner.y_unlabelled = self.Y_oracle
                # self.learner.X_unlabelled = self.X_unlabelled
                classifiers.append(deepcopy(self.learner))

            # Do the active learning!
            while self.X_unlabelled.shape[0] != 0 and not self.stop_function(
                self.learner
            ):

                self.active_learn_iter(classifiers)

        # Write the experiment run results and cleanup intermediate checkpoints
        self._write_run(self.metrics)
        self._cleanup_checkpoint()

        return self.metrics

    def active_learn_iter(self, classifiers):
        # QUERY  ------------------------------------------------------------------------------------------------------------------------------------------
        t_start = time.monotonic()
        if not self.stop_info:
            query_idx, query_points = self.learner.query(self.X_subsampled)
            extra_metrics = {}
        else:
            query_idx, query_points, extra_metrics = self.learner.query(
                self.X_subsampled
            )
        t_elapsed = time.monotonic() - t_start

        # PRE METRICS  -----------------------------------------------------------------------------------------------------------------------------------

        t_ee_start = time.monotonic()
        if any(
            "expected_error" in metric_name if isinstance(metric_name, str) else False
            for metric_name in self.metrics.metrics
        ):
            result = self.ee(
                self.learner, self.X_subsampled, unique_labels=self.unique_labels
            )
            extra_metrics["expected_error_min"] = np.min(result)
            extra_metrics["expected_error_max"] = np.max(result)
            extra_metrics["expected_error_average"] = np.mean(result)
            extra_metrics["expected_error_variance"] = np.var(result)

        extra_metrics["time_ee"] = time.monotonic() - t_ee_start

        if "contradictory_information" in self.metrics.metrics:
            # https://stackoverflow.com/questions/32074239/sklearn-getting-distance-of-each-point-from-decision-boundary
            predictions = self.learner.predict(query_points)
            uncertainties = classifier_uncertainty(self.learner.estimator, query_points)

        # TRAIN  ------------------------------------------------------------------------------------------------------------------------------------------

        if query_points is not None and getattr(
            self.query_strategy, "is_adversarial", False
        ):
            self.learner.teach(query_points, self.Y_oracle[query_idx])

        self.learner.teach(self.X_unlabelled[query_idx], self.Y_oracle[query_idx])

        # POST METRICS  -----------------------------------------------------------------------------------------------------------------------------------

        if "contradictory_information" in self.metrics.metrics:
            contradictory_information = np.sum(
                uncertainties[predictions != self.Y_oracle[query_idx]]
                / np.mean(uncertainties)
            )
            extra_metrics["contradictory_information"] = contradictory_information

        # Replace with non-copying slice?
        if isinstance(self.X_unlabelled, csr_matrix):
            self.X_unlabelled = delete_from_csr(
                self.X_unlabelled, row_indices=query_idx
            )
        else:
            self.X_unlabelled = np.delete(self.X_unlabelled, query_idx, axis=0)

        self.Y_oracle = np.delete(self.Y_oracle, query_idx, axis=0)

        # Resubsample the unlabelled pool. This must happen after we retrain but before metrics are calculated
        # as the subsampled unlabelled pool must be disjoint from the trained instances.
        if self.pool_subsample is not None:
            # TODO: Should this random be seeded?
            self.X_subsampled = self.X_unlabelled[
                np.random.choice(
                    self.X_unlabelled.shape[0],
                    min(self.pool_subsample, self.X_unlabelled.shape[0]),
                    replace=False,
                )
            ]
        else:
            self.X_subsampled = self.X_unlabelled

        extra_metrics["time_total"] = time.monotonic() - t_start

        self.metrics.collect(
            self.metrics.frame.x.iloc[-1] + len(query_idx),
            self.learner.estimator,
            self.Y_test,
            self.X_test,
            time=t_elapsed,  # query time
            X_unlabelled=self.X_subsampled,
            unique_labels=self.unique_labels,
            **extra_metrics,
        )

        if self.ret_classifiers:
            # self.learner.y_unlabelled = self.Y_oracle
            # self.learner.X_unlabelled = self.X_unlabelled
            classifiers.append(self.learner)

        self._checkpoint(self)

    def _checkpoint(self, data):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        for _i in range(3):
            try:
                with open(file, "wb") as f:
                    dill.dump(data, f)
                break
            except Exception:
                print(f"Failed attempt {i+1} of 3 to write to checkpoint {file}")
                pass

    def _restore_checkpoint(self):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        try:
            with open(file, "rb") as f:
                return dill.load(f)
        except FileNotFoundError:
            return None
        except EOFError as e:
            raise Exception(f"Failed to load checkpoint {file}") from e

    def _cleanup_checkpoint(self):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

    def _write_run(self, data):
        file = f"{out_dir()}/runs/{self.config_str}_{self.i}.csv"
        with open(file, "wb") as f:
            pickle.dump(data, f)

    def _restore_run(self):
        file = f"{out_dir()}/runs/{self.config_str}_{self.i}.csv"
        try:
            with open(file, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None


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
        return mat[row_mask][:, col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:, mask]
    else:
        return mat


def expected_error(learner, X, predict_proba=None, p_subsample=1.0, unique_labels=None):
    loss = "binary"

    expected_error = np.zeros(shape=(X.shape[0],))
    possible_labels = (
        unique_labels if unique_labels is not None else np.unique(learner.y_training)
    )

    X_proba = predict_proba or learner.predict_proba(X)

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
                    X_new = data_vstack(
                        (learner.X_training, np.expand_dims(X[x_idx], axis=0))
                    )

                y_new = data_vstack(
                    (
                        learner.y_training,
                        np.array(y).reshape(
                            1,
                        ),
                    )
                )

                cloned_estimator.fit(X_new, y_new)
                refitted_proba = cloned_estimator.predict_proba(X_reduced)

                nloss = _proba_uncertainty(refitted_proba)

                expected_error[x_idx] += np.sum(nloss) * X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] = np.inf

    return expected_error


def csr_vappend(a, b):
    """Takes in 2 csr_matrices and appends the second one to the bottom of the first one.
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data, b.data))
    a.indices = np.hstack((a.indices, b.indices))
    a.indptr = np.hstack((a.indptr, (b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0] + b.shape[0], b.shape[1])
    return a
