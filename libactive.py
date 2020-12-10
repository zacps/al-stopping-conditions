from typing import Tuple, Union, Callable

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from celluloid import Camera
from IPython.core.display import HTML, display
from modAL import batch, density, disagreement, uncertainty, utils
from modAL.models import ActiveLearner, Committee
from sklearn import datasets, metrics, svm, tree
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from libadversarial import fgm, deepfool
from libplot import plot_classification
from libutil import Metrics


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
            estimator=svm.SVC(kernel="linear", probability=True),
            X_training=X_labelled,
            y_training=Y_labelled,
            query_strategy=query_strategy,
        )
    elif model == "committee":
        learner = Committee(
            learner_list=[
                ActiveLearner(
                    estimator=svm.SVC(kernel="linear", probability=True),
                    X_training=X_labelled,
                    y_training=Y_labelled,
                ),
                # committee: logistic regression, svm-linear, svm-rbf, guassian process classifier
                ActiveLearner(
                    estimator=svm.SVC(kernel="rbf", probability=True),
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
) -> Tuple[list, list]:
    """
    Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

    Returns the accuracy curve.
    """

    if model == "svm-linear":
        learner = ActiveLearner(
            estimator=svm.SVC(kernel="linear", probability=True),
            X_training=X_labelled,
            y_training=Y_labelled,
            query_strategy=query_strategy,
        )
    elif model == "committee":
        learner = Committee(
            learner_list=[
                ActiveLearner(
                    estimator=svm.SVC(kernel="linear", probability=True),
                    X_training=X_labelled,
                    y_training=Y_labelled,
                ),
                # committee: logistic regression, svm-linear, svm-rbf, guassian process classifier
                ActiveLearner(
                    estimator=svm.SVC(kernel="rbf", probability=True),
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

        metrics.collect(
            metrics.frame.x.iloc[-1] + len(query_idx), learner.estimator, Y_test, X_test
        )

    return metrics


class MyActiveLearner:
    def __init__(self, animate=False):
        self.animate = animate
        self.metrics = Metrics()
        
        if self.animate:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
            self.cam = Camera(self.fig)
        
    def __setup_learner(self, X_labelled, Y_labelled, query_strategy, model):
        if model == "svm-linear":
            return ActiveLearner(
                estimator=svm.SVC(kernel="linear", probability=True),
                X_training=X_labelled,
                y_training=Y_labelled,
                query_strategy=query_strategy,
            )
        elif model == "committee":
            return Committee(
                learner_list=[
                    ActiveLearner(
                        estimator=svm.SVC(kernel="linear", probability=True),
                        X_training=X_labelled,
                        y_training=Y_labelled,
                    ),
                    # committee: logistic regression, svm-linear, svm-rbf, guassian process classifier
                    ActiveLearner(
                        estimator=svm.SVC(kernel="rbf", probability=True),
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
            
    def __animation_frame(self, learner, X_unlabelled=None, new=None, new_labels=None, start_points=None):
        plot_classification(
            self.ax,
            learner.estimator,
            learner.X_training,
            learner.y_training,
            np.vstack(learner.X_training),
        )
        
        if new is not None:
            self.ax.scatter(new[:,0], new[:,1], cmap=plt.cm.coolwarm, c=new_labels, s=30)
            if start_points is not None:
                for start, end in zip(start_points, new):
                    #self.ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1])
                    self.ax.add_artist(ConnectionPatch(
                        start, end, "data", "data",
                          arrowstyle="->", shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc="w"
                    ))
        if X_unlabelled is not None:
            self.ax.scatter(X_unlabelled[0], X_unlabelled[1], c='black', s=20)
            
        self.ax.text(
            0.9,
            0.05,
            str(
                learner.X_training.shape[0],
            ),
            transform=self.ax.transAxes,
            c="white",
        )
        self.cam.snap()

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
    ) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

        Returns the accuracy curve.
        """

        learner = self.__setup_learner(X_labelled,Y_labelled,query_strategy,model="svm-linear")

        self.metrics.collect(len(X_labelled), learner.estimator, Y_test, X_test)

        if self.animate:
            self.__animation_frame(learner, X_labelled, X_unlabelled)

        while len(X_unlabelled) != 0:
            query_idx, query_points = learner.query(X_unlabelled)
                
            if teach_advesarial and (query_strategy == fgm or query_strategy == deepfool):
                learner.teach(query_points, Y_oracle[query_idx])
                
            learner.teach(X_unlabelled[query_idx], Y_oracle[query_idx])

            X_unlabelled = np.delete(X_unlabelled, query_idx, axis=0)
            Y_oracle = np.delete(Y_oracle, query_idx, axis=0)

            self.metrics.collect(
                self.metrics.frame.x.iloc[-1] + len(query_idx),
                learner.estimator,
                Y_test,
                X_test,
            )

            if self.animate:
                self.__animation_frame(learner, X_labelled, X_unlabelled)

        if self.animate:
            display(
                HTML(cam.animate(interval=500, repeat_delay=1000).to_html5_video())
            )  # milleseconds
            plt.close(fig)

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
    ) -> Tuple[list, list]:
        """
        Perform active learning on the given dataset using a linear SVM model, querying data with the given query strategy.

        Returns metrics.
        """

        learner = self.__setup_learner(X_labelled,Y_labelled,query_strategy,model="svm-linear")

        self.metrics.collect(len(X_labelled), learner.estimator, Y_test, X_test)

        if self.animate:
            self.__animation_frame(learner)

        while not should_stop(learner, self.metrics.frame.iloc[-1]):
            _, query_points, start_points = learner.query(None, learner.X_training, learner.y_training)
            labels = y_oracle(query_points)
            learner.teach(query_points, labels)

            self.metrics.collect(
                self.metrics.frame.x.iloc[-1] + len(query_points),
                learner.estimator,
                Y_test,
                X_test,
            )

            if self.animate:
                self.__animation_frame(learner, new=query_points, new_labels=labels, start_points=start_points)

        if self.animate:
            display(
                HTML(self.cam.animate(interval=500, repeat_delay=1000).to_html5_video())
            )  # milleseconds
            plt.close(self.fig)

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
        clf = svm.SVC(kernel='linear')
        clf.fit(X_labelled+x, Y_labelled+Y_oracle[x_idx])
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
                clf = svm.SVC(kernel='linear')
                
                clf.fit(
                    np.append(np.append(X_labelled, X_unlabelled[x1_idxes], axis=0), [x2], axis=0),
                    np.append(np.append(Y_labelled, Y_oracle[x1_idxes], axis=0), [Y_oracle[x2_idx]], axis=0)
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
        best_idx = np.block([
            [np.array(best_idx), best_idx2.T]
        ])
        
    
    return best_idx