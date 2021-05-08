#!/bin/bash

#poetry run python slurm.py nesi_base.py newsgroups_19 0 1 10-19 --mem 2G --time 1-00:00:00
#poetry run python slurm.py nesi_base.py rcv1_19 1 1 10-19 --mem 4G --time 1-00:00:00
poetry run python slurm.py nesi_base.py webkb_19 2 1 10-19 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_base.py spamassassin_19 3 1 10-19 --mem 2G --time 1-00:00:00
#poetry run python slurm.py nesi_base.py avila_19 4 1 10-19 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_base.py smartphone_19 5 1 10-19 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_base.py swarm_19 6 1 10-19 --mem 4G --time 1-00:00:00
#poetry run python slurm.py nesi_base.py sensorless_19 7 1 10-19 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_base.py splice_19 8 1 10-19 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_base.py anuran_19 9 1 10-19 --mem 612M --time 1-00:00:00

