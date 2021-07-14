from libdatasets import *
import librun
import libstop
from libadversarial import uncertainty_stop
from libstop import rank_stop_conds
from dotenv import load_dotenv

load_dotenv()

matrix = {
    # Dataset fetchers should cache if possible
    # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
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
    "models": [
        "svm-linear",
        # "random-forest",
    ],
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

results = librun.run(matrix, force_cache=True, fragment_run_start=0, fragment_run_end=2)
results_plots = [result[0] for result in results]
classifiers = [result[1] for result in results]
classifiers = [clf for clf in classifiers]

broken = []
for plots, clfs in zip(results_plots, classifiers):
    for i, clfs_ in enumerate(clfs):
        if len(clfs_) != 100:
            broken.append(f"{plots[0].serialize()}_{i}.zip")
if len(broken) != 0:
    print(
        "WARN, some datasets had the wrong number of classifiers:\n" + "\n".join(broken)
    )

from libstop import *

conditions = {
    "GOAL": partial(ZPS, order=2),
    "SSNCut": SSNCut,
    "SC_entropy_mcs": SC_entropy_mcs,
    "SC_oracle_acc": SC_oracle_acc_mcs,
    "SC_mes": SC_mes,
    "Stabilizing Predictions": stabilizing_predictions,
    "Performance Convergence": performance_convergence,
    "Uncertainty Convergence": uncertainty_convergence,
    "Max Confidence": max_confidence,
    "EVM": EVM,
    "VM": VM,
    "Contradictory Information": contradictory_information,
    "Classification Change": classification_change,
    "Overall Uncertainty": overall_uncertainty,
}

stop_conditions, stop_results = libstop.eval_stopping_conditions(
    results_plots, classifiers, conditions=conditions
)
