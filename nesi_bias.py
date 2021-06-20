"""
Entrypoint for NeSI workers.

Takes the following CLI arguments:

"""

import operator
import argparse
from dotenv import load_dotenv

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import librun
from libdatasets import *
from libadversarial import uncertainty_stop
from libactive import delete_from_csr


def bias(
    X_train,
    X_test,
    y_train,
    y_test,
    amount=1e-1,
    rand=None,
    config_str=None,
    i=None,
    **kwargs
):
    """
    Bias data. Find the second most predictive attribute and reduce the prevalence of values above the
    mean for the attribute to amount %. Then, remove the attribute from the test and train data.

    This is supposed to simulate the data being biased by an unknown feature.
    """
    tree = DecisionTreeClassifier(max_depth=1)
    tree.fit(X_train[:1000], y_train[:1000])
    classes = tree.predict(X_train)
    u_classes = np.unique(classes, return_counts=True)

    above_idx = np.where(classes == u_classes[0][np.argmax(u_classes[1])])[0]
    above_idx = rand.choice(above_idx, int(above_idx.shape[0] * amount), replace=False)
    below_idx = np.where(classes != u_classes[0][np.argmax(u_classes[1])])[0]

    X_train = X_train[np.concatenate((above_idx, below_idx))]
    y_train = y_train[np.concatenate((above_idx, below_idx))]

    # X_train = np.delete(X_train, second_most_predictive, axis=1)
    # X_test = np.delete(X_test, second_most_predictive, axis=1)

    # TODO: Shuffle!

    assert X_train.shape[0] == y_train.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert X_test.shape[0] == X_test.shape[0]

    return X_train, X_test, y_train, y_test


matrix = {
    # Dataset fetchers should cache if possible
    # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
    "datasets": [
        # ("newsgroups", wrap(newsgroups, None)),
        ("rcv1", wrap(rcv1, None)),
        # ("webkb", wrap(webkb, None)),
        # ("spamassassin", wrap(spamassassin, None)),
        ("avila", wrap(avila, None)),
        # ("smartphone", wrap(smartphone, None)),
        ("swarm", wrap(swarm, None)),
        ("sensorless", wrap(sensorless, None)),
        # ("splice", wrap(splice, None)),
        ("anuran", wrap(anuran, None)),
    ],
    "dataset_mutators": {"bias2-10": partial(bias, amount=1e-1)},
    "methods": [
        ("uncertainty", partial(uncertainty_stop, n_instances=10)),
    ],
    "models": ["svm-linear"],
    "meta": {
        "dataset_size": 1000,
        "labelled_size": 10,
        "test_size": 0.5,
        "n_runs": 10,
        "ret_classifiers": True,
        "ensure_y": True,
        "stop_info": True,
        "aggregate": False,
        "stop_function": (
            "len1000",
            lambda learner: learner.y_training.shape[0] >= 1000,
        ),
        "pool_subsample": 1000,
    },
}

capture_metrics = [
    accuracy_score,
    f1_score,
    roc_auc_score,
    "time",
    "time_total",
    "time_ee",
    "uncertainty_average",
    "uncertainty_min",
    "uncertainty_max",
    "uncertainty_variance",
    "uncertainty_average_selected",
    "uncertainty_min_selected",
    "uncertainty_max_selected",
    "uncertainty_variance_selected",
    "entropy_max",
    "n_support",
    "contradictory_information",
    "expected_error",
    "expected_error_min",
    "expected_error_max",
    "expected_error_average",
    "expected_error_variance",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment_id", type=int)
    parser.add_argument("fragment_length", type=int)
    parser.add_argument("fragment_run")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    fragment_run = args.fragment_run.split("-")
    start = int(fragment_run[0])
    if len(fragment_run) == 2:
        end = int(fragment_run[1])
    else:
        end = None

    librun.run(
        matrix,
        metrics=capture_metrics,
        # abort=False,
        fragment_id=args.fragment_id,
        fragment_length=args.fragment_length,
        fragment_run_start=start,
        fragment_run_end=end,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    load_dotenv()
    main()
