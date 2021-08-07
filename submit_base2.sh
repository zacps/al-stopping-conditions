#!/bin/bash

poetry run python slurm.py nesi_base2.py rcv1_0_9 1 1 0-9 --mem 4G --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py webkb_0_9 2 1 0-9 --mem 612M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py spamassassin_0_9 3 1 0-9 --mem 2G --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py avila_0_9 4 1 0-9 --mem 612M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py smartphone_0_9 5 1 0-9 --mem 612M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py swarm_0_9 6 1 0-9 --mem 3200M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py sensorless_0_9 7 1 0-9 --mem 612M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py splice_0_9 8 1 0-9 --mem 612M --time 0-00:10:00 --oneshot
poetry run python slurm.py nesi_base2.py anuran_0_9 9 1 0-9 --mem 612M --time 0-00:10:00 --oneshot

