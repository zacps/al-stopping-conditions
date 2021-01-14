from typing import Tuple, Union, Callable
import time

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

from libadversarial import fgm, deepfool
from libplot import plot_classification, plot_poison, c_plot_poison
from libutil import Metrics

# Use GPU-based thundersvm when available
try:
    from thundersvm import SVC
    raise Exception("disabled")
    print("Using ThunderSVM")
except Exception:
    from sklearn.svm import SVC
    print("Using sklearn")


def active_split(X, Y, test_size=0.5, labeled_size=0.1, shuffle=True):
    """
    Split data into three sets:
    * Labeled training set (0.1)
    * Unlabeled training set, to be queried (0.4)
    * Labeled test (0.5)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=shuffle, random_state=42
    )
    X_labelled, X_unlabelled, Y_labelled, Y_oracle = train_test_split(
        X_train,
        Y_train,
        test_size=(1 - labeled_size / test_size),
        shuffle=shuffle,
        random_state=42,
    )

    return X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_test, Y_test


def active_split_query_synthesis(X, Y, test_size=0.5, labeled_size=0.1, shuffle=True):
    """
    Split data into three sets:
    * Labeled training set (0.1)
    * Unlabeled training set, to be queried (0.4)
    * Labeled test (0.5)
    """

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, shuffle=shuffle, random_state=42
    )
    X_labelled, X_unlabelled, Y_labelled, Y_oracle = train_test_split(
        X_train,
        Y_train,
        test_size=(1 - labeled_size / test_size),
        shuffle=shuffle,
        random_state=42,
    )

    return X_labelled, X_unlabelled, Y_labelled, Y_oracle, X_test, Y_test


def active_learn(
    X_labelled,
    X_unlabelled,
    Y_labelled,
    Y_oracle,
    X_test,
    Y_test,
    query_strategy,
    model="svm-linear",
    teach_advesarial=False,
) -> Tuple[list, list]:
    """
    Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

    Returns the accuracy curve.
    """

    if model == "svm-linear":
        learner = ActiveLearner(
            estimator=SVC(kernel="linear", probability=True),
            X_training=X_labelled,
            y_training=Y_labelled,
            query_strategy=query_strategy,
        )
    elif model == "committee":
        learner = Committee(
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

    trained = [len(X_labelled)]
    accuracy = [accuracy_score(Y_test, learner.estimator.predict(X_test))]
    f1 = [f1_score(Y_test, learner.estimator.predict(X_test))]
    roc_auc = [roc_auc_score(Y_test, learner.estimator.decision_function(X_test))]

    while len(X_unlabelled) != 0:
        if query_strategy == fgm:
            query_idx, advesarial_examples = learner.query(X_unlabelled)

            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])
            if teach_advesarial:
                learner.teach(advesarial_examples, Y_oracle[query_idx])
        else:
            query_idx, _ = learner.query(X_unlabelled)
            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])

        X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
        Y_oracle = np.delete(Y_oracle, query_idx, axis=0)

        trained.append(trained[-1] + len(query_idx))

        accuracy.append(accuracy_score(Y_test, learner.estimator.predict(X_test)))
        f1.append(f1_score(Y_test, learner.estimator.predict(X_test)))
        roc_auc.append(
            roc_auc_score(Y_test, learner.estimator.decision_function(X_test))
        )

    return (trained, accuracy, f1, roc_auc)


def active_learn2(
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
) -> Tuple[list, list]:
    """
    Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

    Returns the accuracy curve.
    """

    if model == "svm-linear":
        learner = ActiveLearner(
            estimator=SVC(kernel="linear", probability=True),
            X_training=X_labelled,
            y_training=Y_labelled,
            query_strategy=query_strategy,
        )
    elif model == "committee":
        learner = Committee(
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

    metrics = Metrics()
    metrics.collect(len(X_labelled), learner.estimator, Y_test, X_test)

    while len(X_unlabelled) != 0 and not stop_function(learner):
        if query_strategy == fgm:
            query_idx, advesarial_examples = learner.query(X_unlabelled)

            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])
            if teach_advesarial:
                learner.teach(advesarial_examples, Y_oracle[query_idx])
        else:
            query_idx, _ = learner.query(X_unlabelled)
            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])

        X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
        Y_oracle = np.delete(Y_oracle, query_idx, axis=0)

        metrics.collect(
            metrics.frame.x.iloc[-1] + len(query_idx), learner.estimator, Y_test, X_test
        )

    return metrics


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
    ) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

        Returns the accuracy curve.
        """

        # Take a subset if the unlabelled set size is too large
        if X_unlabelled.shape[0] > 1000:
            print("INFO: Using sample of unlabelled set")
            rng = np.random.default_rng()
            idx = rng.choice(X_unlabelled.shape[0], 1000, replace=False)
            X_unlabelled = X_unlabelled[idx]
            Y_oracle = Y_oracle[idx]

        learner = self.__setup_learner(
            X_labelled, Y_labelled, query_strategy, model="svm-linear"
        )

        self.metrics.collect(len(X_labelled), learner.estimator, Y_test, X_test)

        if self.animate:
            self.__animation_frame(learner, X_unlabelled)

        while len(X_unlabelled) != 0 and not stop_function(learner):
            t_start = time.monotonic()
            query_idx, query_points = learner.query(X_unlabelled)
            t_elapsed = time.monotonic() - t_start

            if query_points is not None and (
                query_strategy == fgm or query_strategy == deepfool
            ):
                learner.teach(query_points, Y_oracle[query_idx])

            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])

            X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
            Y_oracle = np.delete(Y_oracle, query_idx, axis=0)

            self.metrics.collect(
                self.metrics.frame.x.iloc[-1] + len(query_idx),
                learner.estimator,
                Y_test,
                X_test,
                t_elapsed=t_elapsed,
            )

            if self.animate:
                self.__animation_frame(learner, X_unlabelled)

        if self.animate:
            animation = self.cam.animate(interval=500, repeat_delay=1000)
            if self.animation_file is not None:
                animation.save(animation_file)
            display(HTML(animation.to_html5_video()))
            plt.close(self.fig)

        return self.metrics

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


def beam_search(
    X_labelled,
    X_unlabelled,
    Y_labelled,
    Y_oracle,
    X_test,
    Y_test,
):
    scores = []
    for x_idx, x in enumerate(X_unlabelled):
        clf = SVC(kernel="linear")
        clf.fit(X_labelled + x, Y_labelled + Y_oracle[x_idx])
        scores.append(accuracy_score(Y_test, clf.predict(X_test)))
    best_idx = np.argsort(scores)[:5]
    best = X_unlabelled[best_idx]

    best_idx = np.expand_dims(best_idx, axis=-1)

    while True:
        scores = [[], [], [], [], []]
        done_work = False
        for i, x1_idxes in enumerate(best_idx):
            for x2_idx, x2 in enumerate(X_unlabelled):
                if x2 in X_unlabelled[x1_idxes]:
                    continue

                done_work = True
                clf = SVC(kernel="linear")

                clf.fit(
                    np.append(
                        np.append(X_labelled, X_unlabelled[x1_idxes], axis=0),
                        [x2],
                        axis=0,
                    ),
                    np.append(
                        np.append(Y_labelled, Y_oracle[x1_idxes], axis=0),
                        [Y_oracle[x2_idx]],
                        axis=0,
                    ),
                )
                scores[i].append(accuracy_score(Y_test, clf.predict(X_test)))

        if not done_work:
            break

        best_idx2 = np.argsort(scores, axis=None)[:5]

        idx = np.unravel_index(best_idx2, shape=np.array(scores).shape)
        print(best_idx)
        print(scores)
        print(best_idx2)

        if len(idx) == 1:
            idx = idx[0]
        else:
            best_idx2 = idx[1]
        best_idx2 = np.expand_dims(best_idx2, axis=0)
        best_idx = np.block([[np.array(best_idx), best_idx2.T]])

    return best_idx
