"""
Entrypoint for NeSI workers.

Takes the following CLI arguments:

"""

import argparse
import os
from dotenv import load_dotenv

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import librun
from libdatasets import *
from libadversarial import uncertainty_stop

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
        
	# These datasets are not part of 'base', but do provide a large enough change
	# to be considered for inclusion.
	# TODO: Ask Joerg's thoughts on running/including these.
        #("covertype", wrap(covertype, None)),
        #("malware", wrap(malware, None)),
        #("shuttle", wrap(shuttle, None)),
	#("quickdraw", wrap(quickdraw, None)),
    ],
    "dataset_mutators": {
        "none": (lambda *x, **kwargs: x),
    },
    "methods": [
        ("uncertainty", partial(uncertainty_stop, n_instances=10)),
    ],
    "models": [
        "decision-tree"
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
        "stop_function": ("len1000", lambda learner: learner.y_training.shape[0] >= 1000),
        "pool_subsample": 1000
    }
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
    #"n_support",
    "contradictory_information",
    "expected_error",
    "expected_error_min",
    "expected_error_max",
    "expected_error_average",
    "expected_error_variance",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fragment_id', type=int)
    parser.add_argument('fragment_length', type=int)
    parser.add_argument('fragment_run')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--nobackup', action='store_true')

    args = parser.parse_args()

    fragment_run = args.fragment_run.split('-')
    start = int(fragment_run[0])
    if len(fragment_run) == 2:
        end = int(fragment_run[1])
    else:
        end = None

    if args.nobackup:
        os.environ['OUT_DIR'] = "/home/zpul156/out_nobackup"

    librun.run(
        matrix, 
        metrics=capture_metrics,
        #abort=False,
        fragment_id=args.fragment_id,
        fragment_length=args.fragment_length,
        fragment_run_start=start,
        fragment_run_end=end,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    load_dotenv()
    main()