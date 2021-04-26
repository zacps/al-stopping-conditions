#!/bin/bash

poetry run python slurm.py nesi_base.py newsgroups 0 1 10-19 --mem 2G --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py rcv1 1 1 10-19 --mem 4G --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py webkb 2 1 10-19 --mem 612M --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py spamassassin 3 1 10-19 --mem 2G --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py avila 4 1 10-19 --mem 612M --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py smartphone 5 1 10-19 --mem 612M --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py swarm 6 1 10-19 --mem 4G --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py sensorless 7 1 10-19 --mem 612M --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py splice 8 1 10-19 --mem 612M --time 0-00:05:00 --oneshot
poetry run python slurm.py nesi_base.py anuran 9 1 10-19 --mem 612M --time 0-00:05:00 --oneshot

