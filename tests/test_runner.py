from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import librun
from libdatasets import *
from libadversarial import uncertainty_stop


def test_dry_run():
    matrix = {
        # Dataset fetchers should cache if possible
        # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
        "datasets": [
            ("newsgroups", wrap(newsgroups, None)),
            ("rcv1", wrap(rcv1, None)),
            ("webkb", wrap(webkb, None)),
            ("spamassassin", wrap(spamassassin, None)),
            ("avila", wrap(avila, None)),
            ("smartphone", wrap(smartphone, None)),
            ("swarm", wrap(swarm, None)),
            ("sensorless", wrap(sensorless, None)),
            ("splice", wrap(splice, None)),
            ("anuran", wrap(anuran, None)),
        ],
        "dataset_mutators": {
            "none": (lambda *x, **kwargs: x),
        },
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

    librun.run(
        matrix,
        metrics=capture_metrics,
        dry_run=True,
    )
