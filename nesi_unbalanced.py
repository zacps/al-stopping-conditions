"""
Entrypoint for NeSI workers.

Takes the following CLI arguments:

"""

import argparse
import os
from dotenv import load_dotenv

import scipy
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import librun
from libdatasets import *
from libadversarial import uncertainty_stop
from libactive import csr_vappend, active_split

def unbalanced(X_train, X_test, y_train, y_test, amount=1e-1, rand=None, config_str=None, i=None, test_size=None, shuffle=None, **kwargs):
    # Unbalancing might destroy the split, so we undo it and repeat it afterwards
    # Messy, but it should work.
    train_shape = X_train.shape[0]
    test_shape = X_test.shape[0]
    if isinstance(X_train, scipy.sparse.csr_matrix):
        X = csr_vappend(X_train, X_test)
    else:
        X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    unique = np.unique(y)

    idx4 = y==unique[0]
    y4 = y[idx4]
    X4 = X[idx4]
    idx9 = y==unique[1]
    y9 = y[idx9]
    X9 = X[idx9]
    idx = rand.choice(len(y4), int(y4.shape[0]*amount), replace=False)
    X4 = X4[idx]
    y4 = y4[idx]

    if isinstance(X_train, scipy.sparse.csr_matrix):
        X = csr_vappend(X4, X9)
    else:
        X = np.concatenate((X4, X9))
    y = np.concatenate((y4, y9))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand, shuffle=shuffle)

    return X_train, X_test, y_train, y_test

def unbalanced2(X_train, X_test, y_train, y_test, amount=5e-1, rand=None, test_size=None, shuffle=None, **kwargs):
    """
    Amount is the fraction that the majority class should take up in the final data. All other classes are reduced to match
    this proportion.
    """
                    
    # Recalculate class proportions
    class_prop = np.unique(y_train, return_counts=True)

    # Second majority class
    majority = rand.choice(class_prop[0][class_prop[1]==class_prop[1].max()])
    n_in_majority_class = class_prop[1][class_prop[0]==majority]
    
    # Reduce all other classes counts so they make up 1-amount total % of the data,
    # preserving their distribution.
    Xn = [X_train[y_train==majority]]
    yn = [y_train[y_train==majority]]
    for idx, (klass, n) in enumerate(zip(class_prop[0], class_prop[1])):
        if klass == majority:
            continue
            
        this_class_share = n / np.where(y_train != majority)[0].shape[0]
        n_this_class = int(n_in_majority_class*(1-amount)/amount*this_class_share)
        
        klass = np.where(y_train==klass)[0]
        try:
            new_idx = rand.choice(klass, n_this_class, replace=False)
        except ValueError as e:
            print(f"Tried to pick class {n_this_class} instances from {klass.shape[0]}")
            raise e
        Xn.append(X_train[new_idx])
        yn.append(y_train[new_idx])
        
    if isinstance(X_train, scipy.sparse.csr_matrix):
        X_train = scipy.sparse.vstack(Xn)
    else:
        X_train = np.concatenate(Xn)
    y_train = np.concatenate(yn)
     
    # Shuffle train set
    train_idx = rand.choice(y_train.shape[0], y_train.shape[0], replace=False)

    X_train = X_train[train_idx]
    y_train = y_train[train_idx]
        
    return X_train, X_test, y_train, y_test

matrix = {
    # Dataset fetchers should cache if possible
    # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)

    # rcv1, sensorless, anuran are the only datasets to have >3000 instances after being unbalanced
    # maybe a different approach is better? Something non-binary?
    "datasets": [
        #("newsgroups", wrap(newsgroups, None)),
        #("rcv1", wrap(rcv1, None)),
        #("webkb", wrap(webkb, None)),
        #("spamassassin", wrap(spamassassin, None)),
        ("avila", wrap(avila, None)),
        #("smartphone", wrap(smartphone, None)),
        ("swarm", wrap(swarm, None)),
        ("sensorless", wrap(sensorless, None)),
        #("splice", wrap(splice, None)),
        ("anuran", wrap(anuran, None)),
    ],
    "dataset_mutators": {
        "unbalanced2-50": partial(unbalanced2, amount=5e-1)
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
