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
from sklearn.base import clone
from sklearn import calibration
from sklearn.svm import SVC
from modAL.utils.data import data_vstack
from modAL.uncertainty import _proba_uncertainty, classifier_uncertainty
import scipy

from libplot import plot_classification, plot_poison, c_plot_poison
from libutil import Metrics, out_dir

# Use GPU-based thundersvm when available
try:
    from thundersvm import SVC as ThunderSVC
except Exception:
    pass


def active_split(X, Y, test_size=0.5, labeled_size=0.1, shuffle=True, ensure_y=False, random_state=None, mutator=lambda *args, **kwargs: args, config_str=None, i=None):
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
    X_train, X_test, Y_train, Y_test = mutator(X_train, X_test, Y_train, Y_test, rand=random_state, config_str=config_str, i=i)
    
    X_labelled = [] if not isinstance(X, scipy.sparse.csr_matrix) else scipy.sparse.csr_matrix(np.array())
    Y_labelled = []
    
    # ensure a label for all classes made it in to the initial train and validation sets
    for klass in np.unique(Y):
        if klass not in Y_labelled:
            # First value chosen is effectively constant random as the dataset is shuffled
            idx = np.where(Y_oracle==klass)[0][0]
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
        idx = random_state.choice(X_unlabelled.shape[0], labeled_size-X_labelled.shape[0], replace=False)
        
        if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
            X_labelled = csr_vappend(X_labelled, X_unlabelled[idx])
        else:
            X_labelled = np.concatenate((X_labelled, [X_unlabelled[idx]]), axis=0)
        Y_oracle = np.delete(Y_oracle, idx, axis=0)
        if isinstance(X_unlabelled, scipy.sparse.csr_matrix):
            X_unlabelled = delete_from_csr(X_unlabelled, row_indices=[idx])
        else:
            X_unlabelled = np.delete(X_unlabelled, idx, axis=0)
                    
    assert X_labelled.shape[0] == labeled_size
                
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
        X_labelled,
        X_unlabelled,
        Y_labelled,
        Y_oracle,
        X_test,
        Y_test,
        query_strategy,

        stop_function=lambda learner: False,
        ret_classifiers=False,
        stop_info=False,
        config_str=None,
        i=None,
        pool_subsample=None,
        ee='offline',

        model="svm-linear",
        animate=False,
        metrics=None,
        poison=False,
        animation_file=None,
        lb=None,
        ub=None,
    ):
        self.X_labelled = X_labelled
        self.X_unlabelled = X_unlabelled
        self.Y_labelled = Y_labelled
        self.Y_oracle = Y_oracle
        self.X_test = X_test
        self.Y_test = Y_test
        self.query_strategy = query_strategy
        
        self.unique_labels = np.unique(Y_test)
        
        self.stop_function = stop_function
        self.ret_classifiers = ret_classifiers
        self.stop_info = stop_info
        self.config_str = config_str
        self.i = i
        self.pool_subsample = pool_subsample
        self.model = model
        
        self.animate = animate
        self.metrics = Metrics(metrics=metrics)
        self.animation_file = animation_file
        self.poison = poison

        self.lb = lb
        self.ub = ub
        
        if ee == "online":
            self.ee = expected_error_online
        elif ee == "offline":
            self.ee = expected_error
        else:
            raise ValueError(f"ee must be online or offline, got {ee}")

        if self.animate:
            if poison:
                self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 10))
            else:
                self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
            self.cam = Camera(self.fig)

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
            self.metrics.collect(self.X_labelled.shape[0], self.learner.estimator, self.Y_test, self.X_test)
        else:
            self = checkpoint
            
        # Classifiers are stored as a local and explicitly restored as they need to be compressed before being stored.
        with store(f"{out_dir()}/classifiers/{self.config_str}_{self.i}.zip", enable=self.ret_classifiers, restore=checkpoint is not None) as classifiers:
            if self.ret_classifiers and len(classifiers) == 0:
                classifiers.append(deepcopy(self.learner))

            # Initial subsampling, this should probably be done somewhere else tbh...
            if checkpoint is None:
                if self.pool_subsample is not None:
                    # TODO: Should this random be seeded?
                    self.X_subsampled = self.X_unlabelled[np.random.choice(self.X_unlabelled.shape[0], self.pool_subsample, replace=False)] 
                else:
                    self.X_subsampled = self.X_unlabelled

            while self.X_unlabelled.shape[0] != 0 and not self.stop_function(self.learner):
                
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
            query_idx, query_points, extra_metrics = self.learner.query(self.X_subsampled)
        t_elapsed = time.monotonic() - t_start

        # PRE METRICS  -----------------------------------------------------------------------------------------------------------------------------------

        t_ee_start = time.monotonic()
        if any("expected_error" in metric_name if isinstance(metric_name, str) else False for metric_name in self.metrics.metrics):
            result = self.ee(
                self.learner, 
                self.X_subsampled,
                unique_labels=self.unique_labels
            )
            #raise Exception(f"result {result}")
            extra_metrics['expected_error_min'] = np.min(result)
            extra_metrics['expected_error_max'] = np.max(result)
            extra_metrics['expected_error_average'] = np.mean(result)
            extra_metrics['expected_error_variance'] = np.var(result)
        else:
            raise Exception("WHAT?")
        extra_metrics['time_ee'] = time.monotonic() - t_ee_start
            
        if "contradictory_information" in self.metrics.metrics:
            # https://stackoverflow.com/questions/32074239/sklearn-getting-distance-of-each-point-from-decision-boundary
            predictions = self.learner.predict(query_points)
            uncertainties = classifier_uncertainty(self.learner.estimator, query_points)

        # TRAIN  ------------------------------------------------------------------------------------------------------------------------------------------

        if query_points is not None and getattr(self.query_strategy, "is_adversarial", False):
            self.learner.teach(query_points, self.Y_oracle[query_idx])

        self.learner.teach(self.X_unlabelled[query_idx], self.Y_oracle[query_idx])

        # POST METRICS  -----------------------------------------------------------------------------------------------------------------------------------

        if "contradictory_information" in self.metrics.metrics:
            contradictory_information = np.sum(uncertainties[predictions != self.Y_oracle[query_idx]]/np.mean(uncertainties))
            extra_metrics['contradictory_information'] = contradictory_information

        # Replace with non-copying slice?
        if isinstance(self.X_unlabelled, csr_matrix):
            self.X_unlabelled = delete_from_csr(self.X_unlabelled, row_indices=query_idx)
        else:
            self.X_unlabelled = np.delete(self.X_unlabelled, query_idx, axis=0)

        self.Y_oracle = np.delete(self.Y_oracle, query_idx, axis=0)
        
        # Resubsample the unlabelled pool. This must happen after we retrain but before metrics are calculated
        # as the subsampled unlabelled pool must be disjoint from the trained instances.
        if self.pool_subsample is not None:
            # TODO: Should this random be seeded?
            self.X_subsampled = self.X_unlabelled[np.random.choice(self.X_unlabelled.shape[0], min(self.pool_subsample, self.X_unlabelled.shape[0]), replace=False)] 
        else:
            self.X_subsampled = self.X_unlabelled

        self.metrics.collect(
            self.metrics.frame.x.iloc[-1] + len(query_idx),
            self.learner.estimator,
            self.Y_test,
            self.X_test,
            time=t_elapsed,
            X_unlabelled=self.X_subsampled,
            unique_labels=self.unique_labels,
            **extra_metrics
        )

        if self.ret_classifiers:
            classifiers.append(deepcopy(self.learner))

        self._checkpoint(self)
        
    
    def _checkpoint(self, data):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        with open(file, "wb") as f:
            dill.dump(data, f)


    def _restore_checkpoint(self):
        file = f"{out_dir()}/checkpoints/{self.config_str}_{self.i}.pickle"
        try:
            with open(file, "rb") as f:
                return dill.load(f)
        except FileNotFoundError:
            return None


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
    
    
@contextmanager
def store(filename, enable, restore=False):
    if enable:
        inner = CompressedStore(filename, restore=restore)
        try:
            yield inner
        finally:
            inner.close()
    else:
        yield None
        
    
class CompressedStore:
    """
    A compressed, progressively writable object store. Writes individual objects as files in a zip archive.
    Can be read lazily and iterated/indexed as if it were a container.
    
    During writing use the context manager `store` to ensure all changes are reflected.
    """
    
    def __init__(self, filename, restore = False, read = False):
        if read:
            mode = 'r'
        elif restore:
            mode = 'a'
        else:
            mode = 'w'
        self.zip = zipfile.ZipFile(filename, mode, compression=zipfile.ZIP_DEFLATED)
        self.i = len(self.zip.namelist())
        
    def append(self, obj):
        self.zip.writestr(str(self.i), pickle.dumps(obj))
        assert len(self.zip.namelist()) == self.i + 1
        self.i+=1
        
    def __len__(self):
        return self.i
    
    def __getitem__(self, i):
        if isinstance(i, slice):
            if i.start < 0:
                i.start = self.i - i.start
            if i.stop is not None and i.stop < 0:
                i.stop = self.i - i.stop
            try:
                return [pickle.Unpickler(self.zip.open(str(x))).load() for x in range(i.start, i.stop or self.i, i.step or 1)]
            except KeyError:
                raise IndexError(f"index {i} out of range for store of length {self.i}")
            
        if i < 0:
            i = self.i - i
        try:
            return pickle.Unpickler(self.zip.open(str(i))).load()
        except KeyError:
            raise IndexError(f"index {i} out of range for store of length {self.i}")
        
    # As written this is a single use iterable, create a closure here which keeps an ephemeral counter.
    def __iter__(self):
        i = 0
        while i < self.i:
            yield self[i]
            i += 1
        
    def close(self):
        #raise Exception("closed store!")
        self.zip.close()
    
    
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

def expected_error(learner, X, predict_proba=None, p_subsample=1.0, unique_labels=None):
    loss = 'binary'
    
    expected_error = np.zeros(shape=(X.shape[0], ))
    possible_labels = unique_labels if unique_labels is not None else np.unique(learner.y_training)

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
                    X_new = data_vstack((learner.X_training, np.expand_dims(X[x_idx], axis=0)))
            
                y_new = data_vstack((learner.y_training, np.array(y).reshape(1,)))

                cloned_estimator.fit(X_new, y_new)
                refitted_proba = cloned_estimator.predict_proba(X_reduced)
                
                nloss = _proba_uncertainty(refitted_proba)

                expected_error[x_idx] += np.sum(nloss)*X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] = np.inf

    return expected_error


def expected_error_online(learner, X, predict_proba=None, p_subsample=1.0, unique_labels=None):
    loss = 'binary'
    
    expected_error = np.zeros(shape=(X.shape[0], ))
    possible_labels = unique_labels if unique_labels is not None else np.unique(learner.y_training)

    X_proba = predict_proba or learner.predict_proba(X)

    base_estimator = sklearn.linear_model.SGDClassifier(loss='hinge', penalty='l2', eta0=0.1, learning_rate='constant')
    # TODO: Multiple epochs?
    base_estimator.partial_fit(learner.X_training, learner.y_training, classes=possible_labels)

    for x_idx in range(X.shape[0]):
        # subsample the data if needed
        if np.random.rand() <= p_subsample:
            if isinstance(X, csr_matrix):
                X_reduced = delete_from_csr(X, [x_idx])
            else:
                X_reduced = np.delete(X, x_idx, axis=0)
            # estimate the expected error
            for y_idx, y in enumerate(possible_labels):
                #raise Exception(y)
                cloned_estimator = deepcopy(base_estimator)
                cloned_estimator.partial_fit(X[[x_idx]], [y])
                
                #calibrated_estimator = calibration.CalibratedClassifierCV(
                #    # is eta=0.1 right?
                #    # https://stackoverflow.com/questions/23056460/does-the-svm-in-sklearn-support-incremental-online-learning
                #    base_estimator=cloned_estimator,
                #    ensemble=False,
                #    cv='prefit'
                #)
                
                refitted_proba = cloned_estimator.decision_function(X_reduced)
                
                nloss = refitted_proba #_proba_uncertainty(refitted_proba)
                
                #assert (nloss>=0).all() and (nloss<=1).all()

                expected_error[x_idx] += np.sum(np.abs(nloss))*X_proba[x_idx, y_idx]

        else:
            expected_error[x_idx] = np.inf

    assert (expected_error<10000).all() and (expected_error>=0).all()
    return expected_error


def csr_vappend(a,b):
    """ Takes in 2 csr_matrices and appends the second one to the bottom of the first one. 
    Much faster than scipy.sparse.vstack but assumes the type to be csr and overwrites
    the first matrix instead of copying it. The data, indices, and indptr still get copied."""

    a.data = np.hstack((a.data,b.data))
    a.indices = np.hstack((a.indices,b.indices))
    a.indptr = np.hstack((a.indptr,(b.indptr + a.nnz)[1:]))
    a._shape = (a.shape[0]+b.shape[0],b.shape[1])
    return a
