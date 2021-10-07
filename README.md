# Hitting the Target: Stopping Active Learning at the Cost-Based Optimum

This repository contains the code for the paper 'Hitting the Target: Stopping Active Learning at the Cost-Based Optimum'.

## Installation

Necessary dependencies can be installed with:

```bash
$ pip install poetry
$ poetry install
```

## Reproducing

There are four parts to running the results:

1. Configuration
2. Perform the active learning runs
3. Evaluate the stopping criteria
4. Produce the figures and other summary results

### Configuration

Create a `.env` file with two keys, `DATASET_DIR` referring to the location to store the datasets (~`1.4GB`) and `OUT_DIR` referring to the place to record the results (~`1TB` compressed).

For example:

```env
DATASET_DIR=/home/user/datasets
OUT_DIR=/home/user/out
```

### Active Learning Runs

Running the active learning process is time consuming and computationally expensive. For the paper a dedicated 72 core machine was used for the SVM results while the random forest and neural network results were computed on [NeSI](https://nesi.org.nz/).

`nesi_base2.py` is responsible for running active learning, the first parameter is the start of the experiment index to run, the second is the length of experiments to run, and the last is range of seeds (different splits) to run. To run all of the results found in the paper run:

```
$ poetry run nesi_base2.py 0 26 0-30
```

Note that this will likely take upwards of a week even on a powerful machine.

### Evaluating Stopping Criteria

Evaluating stopping criteria given the above results is significantly faster. To evaluate stopping criteria for all of the runs computed in the previous step run:

```
$ poetry run stop_eval.py 0 26 0-30 --jobs=<N_CPUS>
```

Unlike the prior command this does not autodetect the number of CPUs and defaults to 20, so specify an appropriate value for your machine. On a 72 core machine this took approximately three days.

### Produce Summary Figures

To produce the figures and other summary results used in the paper [first register the kernel](https://docs.pymedphys.com/contrib/other/add-jupyter-kernel.html), then start a notebook server:

```
$ jupyter lab
```

From here run `plots_svm.ipynb`, `plots_random_forest.ipynb`, and `plots_neural_network.ipynb` to produce the summary results.

## License

All datasets are the property of their respective owners and are not redistributed with this repository.

Unless otherwise specified all code, including notebooks, are licensed under GPL v2. The text of this license can be found in [LICENSE](LICENSE).
