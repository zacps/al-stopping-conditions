# This file describes the experiment configurations.
# All configurations are modifications of a base 'scaffold' configuration
# which contains the main experiments. Other notable experiments include
# those with different models (random_forest, neural_net), the 100 labelled
# size experiment and the three imperfect data experiments (noise, unbalanced,
# bias).

from functools import partial

from libdatasets import *
from libmutators import *
from libadversarial import uncertainty_stop

SCAFFOLD = {
    "datasets": [
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

BASE = {**SCAFFOLD}

LABELLED100 = {**SCAFFOLD, "meta": {**SCAFFOLD["meta"], "labelled_size": 100}}


RANDOM_FOREST = {
    **SCAFFOLD,
    "models": ["random-forest"],
}


NEURAL_NET = {
    **SCAFFOLD,
    "models": ["neural-network"],
}


BIAS = {
    **SCAFFOLD,
    "datasets": [
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
}

UNBALANCED = {
    **SCAFFOLD,
    # rcv1, sensorless, anuran are the only datasets to have >3000 instances after being unbalanced
    # maybe a different approach is better? Something non-binary?
    "datasets": [
        # ("rcv1", wrap(rcv1, None)),
        # ("webkb", wrap(webkb, None)),
        # ("spamassassin", wrap(spamassassin, None)),
        # ("avila", wrap(avila, None)),
        # ("smartphone", wrap(smartphone, None)),
        ("swarm", wrap(swarm, None)),
        ("sensorless", wrap(sensorless, None)),
        # ("splice", wrap(splice, None)),
        ("anuran", wrap(anuran, None)),
    ],
    "dataset_mutators": {"unbalanced2-70": partial(unbalanced2, amount=7e-1)},
}

# TODO: NOISE
ALL = [BASE, LABELLED100, RANDOM_FOREST, NEURAL_NET, BIAS, UNBALANCED]
