from libactive import active_split
from sklearn.svm import SVC
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.utils import check_random_state
from joblib import Parallel, delayed
import scipy
from libutil import out_dir
import os
import libdatasets
from dotenv import load_dotenv

load_dotenv()


def eval_one(results, name, dataset, run, fname):
    print(f"  {run}")
    if run in results.keys():
        return (run, results[run])

    X, y = dataset()

    X_labelled, X_unlabelled, y_labelled, y_oracle, X_test, y_test = active_split(
        X,
        y,
        labeled_size=10,
        test_size=0.5,
        random_state=check_random_state(run),
        ensure_y=True,
    )
    if isinstance(X_labelled, scipy.sparse.csr_matrix):
        X = scipy.sparse.vstack((X_labelled, X_unlabelled))
    else:
        X = np.concatenate((X_labelled, X_unlabelled))
    y = np.concatenate((y_labelled, y_oracle))

    clf = SVC(probability=True, kernel="linear")
    clf.fit(X, y)
    predicted = clf.predict(X_test)
    predict_proba = clf.predict_proba(X_test)
    unique_labels = np.unique(y_labelled)

    if len(unique_labels) > 2 or len(unique_labels.shape) > 1:
        roc_auc = roc_auc_score(y_test, predict_proba, multi_class="ovr")
    else:
        roc_auc = roc_auc_score(y_test, predict_proba[:, 1])

    result = [
        accuracy_score(y_test, predicted),
        f1_score(
            y_test,
            predicted,
            average="micro" if len(unique_labels) > 2 else "binary",
            pos_label=unique_labels[1] if len(unique_labels) <= 2 else 1,
        ),
        roc_auc,
    ]

    return (run, result)


def run_passive(datasets, runs):
    all_results = {}
    for name, dataset in datasets:
        if name == "newsgroups":
            continue
        print(name)
        fname = f"{out_dir()}{os.path.sep}passive{os.path.sep}{name}.pickle"
        try:
            with open(fname, "rb") as f:
                results = pickle.load(f)
                print(f"Have results for {name}")
                if all([run in results.keys() for run in runs]):
                    all_results[name] = results
                    continue
        except (FileNotFoundError, EOFError):
            results = {}

        # os.cpu_count()
        r = Parallel(n_jobs=min(os.cpu_count(), len(runs)))(
            delayed(eval_one)(results, name, dataset, run, fname) for run in runs
        )
        for run, result in r:
            results[run] = result
        with open(fname, "wb") as f:
            pickle.dump(results, f)
        all_results[name] = results
    return all_results


from nesi_noise import matrix


def key(dataset):
    return dataset[1]()[0].shape[0]


datasets = sorted(matrix["datasets"], key=key)

datasets = [datasets[-1]]

run_passive(datasets, range(10))
