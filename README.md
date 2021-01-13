# Adversarial Active Learning

This repository contains the code from a summer research project at the University of Auckland.

The early draft of the report can be found [here](https://www.overleaf.com/read/yvghhwrnpvmk).

## Installation

```bash
$ pip install poetry
$ poetry install
```

This project will make use of [ThunderSVM](https://github.com/Xtra-Computing/thundersvm/) if available. See their repository for installation instructions.

## Reproducing

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

All datasets are the property of their respective owners.

Unless otherwise specified all code, including notebooks, are licensed under GPL. The text of this license can be found in [LICENSE]().
