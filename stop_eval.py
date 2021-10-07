from libstop import *
import argparse
import os
import logging
from libdatasets import *
import librun
import libstop
from libadversarial import uncertainty_stop
from libstop import rank_stop_conds
from dotenv import load_dotenv
import time
import datetime

logger = logging.getLogger(__name__)

load_dotenv()

matrix = {
    # Dataset fetchers should cache if possible
    # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
    "datasets": [
        ("rcv1-58509", wrap(rcv1, 58509)),
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
            "res500",
            lambda learner, matrix, state: state.X_unlabelled.shape[0] < 510,
        ),
        "pool_subsample": 1000,
    },
}


def main(args):
    fragment_run = args.fragment_run.split("-")
    start = int(fragment_run[0])
    if len(fragment_run) == 2:
        end = int(fragment_run[1])
    else:
        end = None

    s = time.monotonic()
    results = librun.run(
        matrix,
        force_cache=True,
        fragment_id=args.fragment_id,
        fragment_length=args.fragment_length,
        fragment_run_start=start,
        fragment_run_end=end,
    )
    results_plots = [result[0] for result in results]
    classifiers = [result[1] for result in results]
    classifiers = [clf for clf in classifiers]
    print(
        "Retrieving classifier results took:",
        str(datetime.timedelta(seconds=time.monotonic() - s)),
    )

    conditions = {
        "SSNCut": SSNCut,
        # "SC_entropy_mcs": SC_entropy_mcs,
        "SC_mes": SC_mes,
        # "SC_oracle_acc": SC_oracle_acc_mcs,
        # "Stabilizing Predictions": StabilizingPredictions,
        # "Performance Convergence": PerformanceConvergence,
        # "Uncertainty Convergence": UncertaintyConvergence,
        # "Max Confidence": MaxConfidence,
        # "EVM": EVM,
        # "VM": VM,
        # "Contradictory Information": ContradictoryInformation,
        "Classification Change": ClassificationChange,
        # "Overall Uncertainty": OverallUncertainty,
    }

    s = time.monotonic()
    stop_conditions, stop_results = libstop.eval_stopping_conditions(
        results_plots,
        classifiers,
        conditions=conditions,
        recompute=["SC_mes", "Classification Change", "SSNCut"],
        jobs=args.jobs,
    )
    print(
        "Computing stop conditions took:",
        str(datetime.timedelta(seconds=time.monotonic() - s)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fragment_id", type=int)
    parser.add_argument("fragment_length", type=int)
    parser.add_argument("fragment_run")
    parser.add_argument("--jobs", type=int, default=20)

    args = parser.parse_args()
    main(args)
