#!/bin/bash

#poetry run python slurm.py nesi_nn.py newsgroups-nn_29 0 1 2-9 --mem 2G --time 1-00:00:00
#poetry run python slurm.py nesi_nn.py rcv1-nn_29 1 1 2-9 --mem 4G --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py webkb-nn_29 2 1 2-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py spamassassin-nn_29 3 1 2-9 --mem 2G --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py avila-nn_29 4 1 2-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py smartphone-nn_29 5 1 2-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py swarm-nn_29 6 1 2-9 --mem 4G --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py sensorless-nn_29 7 1 2-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py splice-nn_29 8 1 2-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_nn.py anuran-nn_29 9 1 2-9 --mem 612M --time 0-00:02:00 --oneshot

#poetry run python slurm.py nesi_nn.py spamassassin-nn_01 3 1 0-1 --mem 2G --time 1-00:00:00
#poetry run python slurm.py nesi_nn.py smartphone-nn_01 5 1 0-1 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_nn.py sensorless-nn_01 7 1 0-1 --mem 612M --time 1-00:00:00

# --------------------------

#poetry run python slurm.py nesi_nn.py avila-nn_29 4 1 2-9 --mem 612M --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py sensorless-nn_29 7 1 2-9 --mem 612M --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py splice-nn_29 8 1 2-9 --mem 612M --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py anuran-nn_29 9 1 2-9 --mem 612M --time 0-01:00:00

# To Restart:
poetry run python slurm.py nesi_nn.py rcv1-nn_29 1 1 2-9 --mem 4G --time 0-04:00:00
#poetry run python slurm.py nesi_nn.py webkb-nn_29 2 1 2-9 --mem 612M --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py spamassassin-nn_29 3 1 2-9 --mem 2G --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py swarm-nn_29 6 1 2-9 --mem 4G --time 0-01:00:00
#poetry run python slurm.py nesi_nn.py smartphone-nn_29 5 1 2-9 --mem 612M --time 0-01:00:00
