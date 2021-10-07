# Hitting the Target: Stopping Active Learning at the Cost-Based Optimum

This repository contains the code for the paper 'Hitting the Target: Stopping Active Learning at the Cost-Based Optimum'.

## Installation

Necesssary dependencies can be installed with:

```bash
$ pip install poetry
$ poetry install
```

## Reproducing

There are three parts to running the results:

1. Perform the active learning runs
2. Evaluate the stopping criteria
3. Produce the figures and other summary results

### Active Learning Runs

Running the active learning process is time consuming and computationally expensive. For the paper a dedicated 72 core machine was used for the SVM results while the neural net.

```bash
$ poetry run jupyter notebook
```

As this repository contains the complete code used to produce the results in the report it should be as simple as running the notebook containing the results you are interested in. Note, the runner framework `librun.run` caches results by default which are checked in. Hence to run the experiments yourself you will need to pass `force_run=True`.

## Structure

* [libactive.py]() contains `MyActiveLearner` which runs a single active
  learning experiment and measures metrics over its duration.
* [libadversarial.py]() contains active learning query strategies, both
  pool-based and query-synthesis. Many of these make use of adversarial 
  attacks
* [libdatasets.py]() contains dataset fetchers. Most of the datasets come
  from the UCI machine learning repository.
* [libplot.py]() contains functions for plotting decision boundaries, attack
  objective functions, and more.
* [librun.py]() contains the multi-experiment runner responsible for
  parallelism, repeated measurements, and the run configuration matrix.
* [libutil.py]() contains the metrics measurement class and other convencience
  classes.

## License

All datasets are the property of their respective owners and are not redistributed with this repository.

Unless otherwise specified all code, including notebooks, are licensed under GPL v2. The text of this license can be found in [LICENSE](LICENSE).
