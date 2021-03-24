"""
Entrypoint for NeSI workers.

Takes the following CLI arguments:

"""

import argparse
from dotenv import load_dotenv

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import librun
from libdatasets import *
from libadversarial import uncertainty_stop

matrix = {
    # Dataset fetchers should cache if possible
    # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
    "datasets": [
        # Text classification
        
        # https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.31.6090&rep=rep1&type=pdf
        ("newsgroups_faith", wrap(newsgroups, None, ('alt.atheism', 'soc.religion.christian'))),
        ("newsgroups_graphics", wrap(newsgroups, None, ('comp.graphics', 'comp.windows.x'))),
        ("newsgroups_hardware", wrap(newsgroups, None, ('comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'))),
        ("newsgroups_sports_crypto", wrap(newsgroups, None, ('rec.sport.baseball', 'sci.crypt'))),
    
        ("rcv1", wrap(rcv1, None)),
        ("webkb", wrap(webkb, None)),
        ("spamassassin", wrap(spamassassin, None)),
        
        # Image classification
        ("cifar10", wrap(cifar10, None)),
        ("quickdraw", wrap(quickdraw, None)),
        ("avila", wrap(avila, None)),
        
        # General
        ("shuttle", wrap(shuttle, None)),
        #("covertype", wrap(covertype, None)), # fit takes a million years (1233s for 1000 instances)
        ("smartphone", wrap(smartphone, None)),
        ("htru2", wrap(htru2, None)),
        #("malware", wrap(malware, None)), # MALWARE FIT DID NOT FINISH (07:30:30.xxx CPU time)
        ("bidding", wrap(bidding, None)),
        ("swarm", wrap(swarm, None)),
        ("bank", wrap(bank, None)),
        ("buzz", wrap(buzz, None)), # Slow fit times
        ("sensorless", wrap(sensorless, None)),
        ("dota2", wrap(dota2, None)),
        
        # Bio
        ("abalone", wrap(abalone, None)),
        ("splice", wrap(splice, None)),
        ("anuran", wrap(anuran, None)),
        
        # Medical
        ("cardio", wrap(cardio, None)),
        ("skin", wrap(skin, None)),
        
    ],
    "dataset_mutators": {
        "none": (lambda *x, **kwargs: x),
    },
    "methods": [
        ("uncertainty", partial(uncertainty_stop, n_instances=10)),
    ],
    "models": [
        "svm-linear"
    ],
    "meta": {
        "dataset_size": 1000,
        "labelled_size": 10,
        "test_size": {
            "newsgroups_faith": 500,
            "newsgroups_graphics": 500,
            "newsgroups_hardware": 500,
            "newsgroups_sports_crypto": 500,
            "*": 0.5
        },
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
    parser.add_argument('fragment_id', type=int)
    parser.add_argument('fragment_length', type=int)
    parser.add_argument('fragment_run')
    parser.add_argument('--dry-run', action='store_true')

    args = parser.parse_args()

    fragment_run = args.fragment_run.split('-')
    start = int(fragment_run[0])
    if len(fragment_run) == 2:
        end = int(fragment_run[1])
    else:
        end = None

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
