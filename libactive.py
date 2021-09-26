import time
import pickle
import os
import json
import zipfile
import logging
from copy import deepcopy
from typing import Tuple, Callable
from contextlib import contextmanager

import sklearn
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from celluloid import Camera

try:
    from IPython.core.display import display
except ModuleNotFoundError:
    pass
from modAL import disagreement
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
from sklearn.metrics.pairwise import euclidean_distances
from modAL.utils.data import data_vstack
from modAL.uncertainty import _proba_uncertainty, classifier_uncertainty
import scipy

from libutil import Metrics, atomic_write, out_dir
from libstore import store, CompressedStore
from libconfig import Config
from modal_learner import IndexLearner


logger = logging.getLogger(__name__)


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
        # Varying sets
        self._initial_X_labelled = X_labelled
        self._initial_X_unlabelled = X_unlabelled
        self._initial_Y_labelled = Y_labelled
        self._initial_Y_oracle = Y_oracle

        # Associated index sets
        self._taught_idx = np.array([], dtype=int)

        # Constant sets
        self.X_test = X_test
        self.Y_test = Y_test

        self.query_strategy = query_strategy

        self.unique_labels = np.unique(Y_test)

        stop_func = config.meta.get("stop_function", ("default", lambda learner: False))
        self.stop_function = stop_func[1]
        self.stop_function_name = stop_func[0]
        self.ret_classifiers = config.meta.get("ret_classifiers", False)
        self.stop_info = config.meta.get("stop_info", False)
        self.config_str = config.serialize()
        self.i = i
        self.pool_subsample = config.meta.get("pool_subsample", None)
        self.model = config.model_name.lower()
        self.dataset_name = config.dataset_name

        self.cleanup_checkpoints = True

        self.metrics = Metrics(metrics=metrics)

        self.iteration = 0

        # Configuration string for a previous run
        self.config_str_1000 = self.config_str.replace(
            self.stop_function_name, "len1000"
        )

        # Validate expected error method
        ee = config.meta.get("ee", "offline")
        if ee == "online":
            self.ee = expected_error_online
        elif ee == "offline":
            self.ee = expected_error
        else:
            raise ValueError(f"ee must be online or offline, got {ee}")

    def __setup_learner(self):
        kwargs = {
            "X_training": self.X_labelled,
            "y_training": self.Y_labelled,
            "X_unlabelled": self._initial_X_unlabelled,
            "y_unlabelled": self._initial_Y_oracle,
            "query_strategy": self.query_strategy,
        }
        if self.model == "svm-linear":
            return IndexLearner(
                estimator=SVC(kernel="linear", probability=True), **kwargs
            )
        elif self.model == "thunder-svm-linear":
            return IndexLearner(
                estimator=ThunderSVC(kernel="linear", probability=True), **kwargs
            )
        elif self.model == "svm-rbf":
            return IndexLearner(estimator=SVC(kernel="rbf", probability=True), **kwargs)
        elif self.model == "svm-poly":
            return IndexLearner(
                estimator=SVC(kernel="poly", probability=True), **kwargs
            )
        elif self.model == "random-forest":
            return IndexLearner(estimator=RandomForestClassifier(), **kwargs)
        elif self.model == "decision-tree":
            return IndexLearner(estimator=DecisionTreeClassifier(), **kwargs)
        elif self.model == "gaussian-nb":
            return IndexLearner(estimator=GaussianNB(), **kwargs)
        elif self.model == "k-neighbors":
            return IndexLearner(estimator=KNeighborsClassifier(), **kwargs)
        elif self.model == "perceptron":
            return IndexLearner(estimator=Perceptron(), **kwargs)
        elif self.model == "neural-network":
            return IndexLearner(
                estimator=MLPClassifier(
                    hidden_layer_sizes=(100,),  # default
                    activation="relu",  # default
                ),
                **kwargs,
            )
        else:
            raise Exception("unknown model")

    def stop_function_adapter(self, learner, *args):
        "Adapter which allows stop functions which don't accept *args"
        try:
            return self.stop_function(learner, *args)
        except TypeError:
            return self.stop_function(learner)

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
            past_run_1000 = self.try_restore_1000()
            if past_run_1000 is not None:
                print("Restoring from previous 1000 instance run")
                self = past_run_1000
                # Required so we don't overwrite the classifier file below!
                checkpoint = True
            else:
                self.learner = self.__setup_learner()
                self.metrics.collect(
                    self.X_labelled.shape[0],
                    self.learner.estimator,
                    self.Y_test,
                    self.X_test,
                )
        else:
            print("Restoring from checkpoint")
            # FIXME: Does this actually work how I expect...?
            self.__dict__ = checkpoint.__dict__
            # self = checkpoint

        # Classifiers are stored as a local and explicitly restored as they need to be
        # compressed before being stored.
        with store(
            f"{out_dir()}/classifiers/{self.config_str}_{self.i}.zip",
            # References to these are held in stores
            self._initial_X_labelled,
            self._initial_X_unlabelled,
            self._initial_Y_labelled,
            self._initial_Y_oracle,
            enable=self.ret_classifiers,
            restore=checkpoint is not None,
        ) as classifiers:
            # Initial subsampling, this should probably be done somewhere else tbh...
            if checkpoint is None:
                self._update_subsample()

            if self.ret_classifiers and len(classifiers) == 0:
                classifiers.append(deepcopy(self.learner))

            # Do the active learning!
            # Need to make sure we check the stop function before writing to any result files
            # in case we restore from a past 1000 checkpoint (because we no longer delete them)
            # even though we don't need to do any more work.

            # TODO: Change mandatory check to >= 500 if we keep the reserve?
            while self.X_unlabelled.shape[0] >= 10 and not self.stop_function_adapter(
                self.learner, self.metrics, self
            ):

                self.active_learn_iter(classifiers)

        # Write the experiment run results and (possibly) cleanup intermediate checkpoints
        self._write_run(self.metrics)
        self._cleanup_checkpoint()

        return self.metrics

    def active_learn_iter(self, classifiers):
        print(f"Starting iteration {self.iteration}")
        # QUERY  -------------------------------------------------------------------------------------
        t_start = time.monotonic()
        query_idx, query_points, extra_metrics = self.learner.query(self.X_subsampled)
        t_elapsed = time.monotonic() - t_start

        # PRE METRICS  -------------------------------------------------------------------------------

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

        # TRAIN  -------------------------------------------------------------------------------------

        # This has the effect of applying the subsampled index to the unlabelled pool then
        # applying the query index
        self.learner.teach(self._index_X_subsampled[query_idx])

        # POST METRICS  ------------------------------------------------------------------------------

        if "contradictory_information" in self.metrics.metrics:
            contradictory_information = np.sum(
                uncertainties[predictions != self.Y_oracle[query_idx]]
                / np.mean(uncertainties)
            )
            extra_metrics["contradictory_information"] = contradictory_information

        # Update taught instances
        self._taught_idx = np.concatenate(
            (self._taught_idx, self._index_X_subsampled[query_idx])
        )

        # Resubsample the unlabelled pool. This must happen after we retrain but before metrics are calculated
        # as the subsampled unlabelled pool must be disjoint from the trained instances.
        self._update_subsample()

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
            classifiers.append(self.learner)

        self.iteration += 1

        self._checkpoint(self)

    def _update_subsample(self):
        mask = np.ones(self._initial_X_unlabelled.shape[0], dtype=bool)
        mask[self._taught_idx] = False
        indexes = np.where(mask)[0]
        if self.pool_subsample is not None:
            self._index_X_subsampled = np.random.choice(
                indexes,
                min(self.pool_subsample, self.X_unlabelled.shape[0]),
                replace=False,
            )
        else:
            self._index_X_subsampled = indexes

    @property
    def X_subsampled(self):
        return self._initial_X_unlabelled[self._index_X_subsampled]

    @property
    def y_subsampled(self):
        return self._initial_y_unlabelled[self._index_X_subsampled]

    @property
    def X_unlabelled(self):
        mask = np.ones(self._initial_X_unlabelled.shape[0], dtype=bool)
        mask[self._taught_idx] = False
        return self._initial_X_unlabelled[mask]

    @property
    def Y_oracle(self):
        mask = np.ones(self._initial_X_unlabelled.shape[0], dtype=bool)
        mask[self._taught_idx] = False
        return self._initial_Y_oracle[mask]

    @property
    def X_labelled(self):
        if isinstance(self._initial_X_labelled, csr_matrix):
            return scipy.sparse.vstack(
                (self._initial_X_labelled, self._initial_X_unlabelled[self._taught_idx])
            )
        else:
            return data_vstack(
                (self._initial_X_labelled, self._initial_X_unlabelled[self._taught_idx])
            )

    @property
    def Y_labelled(self):
        if isinstance(self._initial_Y_labelled, csr_matrix):
            return scipy.sparse.vstack(
                (self._initial_Y_labelled, self._initial_Y_oracle[self._taught_idx])
            )
        else:
            return data_vstack(
                (self._initial_Y_labelled, self._initial_Y_oracle[self._taught_idx])
            )

    def _checkpoint(self, data):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        for i in range(3):
            try:
                # Guard against writing partial checkpoints by using an atomic copy.
                with atomic_write(file, "wb") as f:
                    dill.dump(data, f)
                break
            except Exception as e:
                print(f"Failed attempt {i+1} of 3 to write to checkpoint {file}: {e}")
                pass

    def try_restore_1000(self):
        "Try to restore from a previous run that was terminated at 1000 instances."
        # DISABLED
        if True:
            return None

        # This is a bad way to do it, but it is what it is. We only have ~5 spare characters
        # in the result filenames.
        if self.stop_function_name == "len1000":
            return None

        try:
            # If in future we need to restore again from runs this is where the test needs to happen
            # though in this case we should have checkpoints (as we don't remove them for resumed runs)
            # so it'll be a bit easier (& faster).
            print(f"Reading past result from {self.config_str_1000} ({self.i})")
            cached_config, metrics_list = self._read_result(
                self.config_str_1000, runs=[self.i]
            )
            metrics = metrics_list[0]  # first and only requested run
            classifiers = self.__read_classifiers(self.config_str_1000, i=self.i)
        except FileNotFoundError:
            print("Could not restore from 1000 instance run")
            return None

        assert (
            len(classifiers) == 100
        ), f"Could not restore from 1000 instance run as it has {len(classifiers)}!=100 iterations"
        assert (
            len(metrics.x) == 100
        ), f"Could not restore from 1000 instance run as it has {len(metrics.x)}!=100 iterations"

        self.metrics.frame = metrics
        dense_atol = 1e-1 if self.dataset_name == "swarm" else 1e-3
        self.X_unlabelled, self.Y_oracle = reconstruct_last_unlabelled(
            classifiers, self.X_unlabelled, self.Y_oracle, dense_atol=dense_atol
        )
        self.learner = classifiers[-1]

        classifiers.close()

        self._update_subsample()
        self.cleanup_checkpoints = False

        return self

    def _read_result(self, config_str, runs):
        results = []
        for name in [f"{out_dir()}{os.path.sep}{config_str}_{i}.csv" for i in runs]:
            with open(name, "r") as f:
                cached_config = Config(
                    **{"model_name": "svm-linear", **json.loads(f.readline())}
                )
                results.append(pd.read_csv(f, index_col=0))
        # make the run numbers available
        cached_config.runs = runs
        return cached_config, results

    def __read_classifiers(self, config_str, i):
        zfile = f"{out_dir()}{os.path.sep}classifiers{os.path.sep}{config_str}_{i}.zip"

        return CompressedStore(zfile, read=True)

    def _restore_checkpoint(self):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        try:
            with open(file, "rb") as f:
                my_active_learner = dill.load(f)
                my_active_learner.learner._X_training = self._initial_X_labelled
                my_active_learner.learner._y_training = self._initial_Y_labelled
                my_active_learner.learner._X_unlabelled = self._initial_X_unlabelled
                my_active_learner.learner._y_unlabelled = self._initial_Y_oracle
                return my_active_learner
        except FileNotFoundError:
            return None
        except EOFError as e:
            raise Exception(f"Failed to load checkpoint {file}") from e

    def _cleanup_checkpoint(self):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        if self.cleanup_checkpoints:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        else:
            print(f"Cleaning checkpoints disabled, not removing {file}")

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


def reconstruct_last_unlabelled(clfs, X_unlabelled, Y_oracle, dense_atol=1e-6):
    """
    Reconstruct the last unlabelled pool from stored information.

    This is used to resume experiments that were terminated at 1000 instances.
    """
    # Defensive asserts
    assert clfs is not None, "Classifiers must be non-none"
    assert len(clfs) == 100
    assert X_unlabelled is not None, "X_unlabelled must be non-none"
    assert Y_oracle is not None, "Y_oracle must be non-none"
    assert (
        X_unlabelled.shape[0] == Y_oracle.shape[0]
    ), "unlabelled and oracle pools have a different shape"

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
            pairs = np.where(np.isclose(dists, 0, atol=1e-1, rtol=0))
            pairs = np.array(
                [
                    [A_idx, B_idx]
                    for A_idx, B_idx in zip(*pairs)
                    if (A[A_idx] == B[B_idx]).all()
                ]
            ).T

        return pairs

    clf = clfs[-1]
    assert clf.X_training.shape[0] == 1000, f"{clf.X_training.shape[0]} == 1000"

    equal_rows = list(
        compare(
            X_unlabelled,
            clf.X_training,
            sparse=isinstance(X_unlabelled, scipy.sparse.csr_matrix),
        )
    )

    # Unlike in reconstruct_unlabelled we are doing this in one shot, and are looking
    # for all instances which need to be removed. This is eqaul to 1000 minus the
    # initial set size which were never present in the unlabelled pool.
    target_n = 1000 - clfs[0].X_training.shape[0]

    # Some datasets (rcv1) contain duplicates. These were only queried once, so we
    # make sure we only remove a single copy from the unlabelled pool.
    if len(equal_rows[0]) > target_n:
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

    elif len(equal_rows[0]) == target_n:
        # Fast path with no duplicates
        assert (Y_oracle[equal_rows[0]] == clf.y_training[equal_rows[1]]).all()
        really_equal_rows = equal_rows[0]
    else:
        raise Exception(
            f"Less than {target_n} ({len(equal_rows[0])}) equal rows were found."
            + " This could indicate an issue with the row-wise compare"
            + " function."
        )

    assert (
        len(really_equal_rows) == target_n
    ), f"{len(really_equal_rows)}=={target_n} (target)"

    n_before = X_unlabelled.shape[0]
    if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
        X_unlabelled = delete_from_csr(X_unlabelled, really_equal_rows)
    else:
        X_unlabelled = np.delete(X_unlabelled, really_equal_rows, axis=0)
    Y_oracle = np.delete(Y_oracle, really_equal_rows, axis=0)
    assert (
        X_unlabelled.shape[0] == n_before - target_n
    ), f"We found 10 equal rows but {n_before-X_unlabelled.shape[0]} were removed"

    return (X_unlabelled.copy(), Y_oracle)
