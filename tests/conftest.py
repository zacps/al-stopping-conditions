import os
from tempfile import TemporaryDirectory

import pytest
from dotenv import load_dotenv, dotenv_values

import librun
from libdatasets import *
from libadversarial import uncertainty_stop


@pytest.fixture
def results():
    "Load results to test stopping conditions."
    load_dotenv()

    matrix = {
        # Dataset fetchers should cache if possible
        # Lambda wrapper required for function to be pickleable (sent to other threads via joblib)
        "datasets": [
            ("avila", wrap(avila, None)),
        ],
        "dataset_mutators": {
            "none": (lambda *x, **kwargs: x),
        },
        "methods": [
            ("uncertainty", partial(uncertainty_stop, n_instances=10)),
        ],
        "models": [
            "svm-linear",
            "decision-tree",
            "random-forest",
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

    results = librun.run(
        matrix, force_cache=True, fragment_run_start=0, fragment_run_end=1
    )
    results_plots = [result[0] for result in results]
    classifiers = [result[1] for result in results]
    classifiers = [clf for clf in classifiers]

    for plots, clfs in zip(results_plots, classifiers):
        for i, clfs_ in enumerate(clfs):
            if len(clfs_) != 100:
                raise Exception(f"{plots[0].serialize()}_{i}.zip")

    del os.environ["DATASET_DIR"]
    del os.environ["OUT_DIR"]

    return results_plots, classifiers


@pytest.fixture
def dataset_dir():
    os.environ["DATASET_DIR"] = dotenv_values()["DATASET_DIR"]
    yield
    del os.environ["DATASET_DIR"]


@pytest.fixture
def temp_out():
    with TemporaryDirectory() as tempdir:
        os.environ["OUT_DIR"] = tempdir
        os.mkdir(f"{tempdir}/stopping")
        os.mkdir(f"{tempdir}/classifiers")
        os.mkdir(f"{tempdir}/checkpoints")
        os.mkdir(f"{tempdir}/passive")
        os.mkdir(f"{tempdir}/runs")
        yield
        del os.environ["OUT_DIR"]
