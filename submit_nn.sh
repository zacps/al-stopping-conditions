#!/bin/bash

poetry run python slurm.py nesi_nn.py rcv1-nn 1 1 0-1 --mem 4G --time 1-00:00:00
poetry run python slurm.py nesi_nn.py swarm-nn 6 1 0-1 --mem 4G --time 1-00:00:00
poetry run python slurm.py nesi_nn.py newsgroups-nn 0 1 0-1 --mem 2G --time 1-00:00:00
poetry run python slurm.py nesi_nn.py spamassassin-nn 3 1 0-1 --mem 2G --time 1-00:00:00
poetry run python slurm.py nesi_nn.py avila-nn 4 1 0-1 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_nn.py smartphone-nn 5 1 0-1 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_nn.py webkb-nn 2 1 0-1 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_nn.py sensorless-nn 7 1 0-1 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_nn.py splice-nn 8 1 0-1 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_nn.py anuran-nn 9 1 0-1 --mem 612M --time 1-00:00:00

