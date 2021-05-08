#!/bin/bash

#poetry run python slurm.py nesi_labelled_size.py newsgroups-labelled_69 0 1 6-9 --mem 2G --time 1-00:00:00
#poetry run python slurm.py nesi_labelled_size.py avila-labelled_69 4 1 6-9 --mem 612M --time 0-00:02:00 --oneshot

#poetry run python slurm.py nesi_labelled_size.py avila-labelled_69 4 1 6-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_labelled_size.py rcv1-labelled_69 1 1 6-9 --mem 4G --time 0-00:02:00 --nobackup 

poetry run python slurm.py nesi_labelled_size.py rcv1-labelled_69 1 1 6-9 --mem 4G --time 1-00:00:00 --nobackup
poetry run python slurm.py nesi_labelled_size.py webkb-labelled_69 2 1 6-9 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_labelled_size.py spamassassin-labelled_69 3 1 6-9 --mem 2G --time 1-00:00:00
poetry run python slurm.py nesi_labelled_size.py avila-labelled_69 4 1 6-9 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_labelled_size.py smartphone-labelled_69 5 1 6-9 --mem 612M --time 1-00:00:00 --nobackup
poetry run python slurm.py nesi_labelled_size.py swarm-labelled_69 6 1 6-9 --mem 4G --time 1-00:00:00 --nobackup
poetry run python slurm.py nesi_labelled_size.py sensorless-labelled_69 7 1 6-9 --mem 612M --time 1-00:00:00 --nobackup
poetry run python slurm.py nesi_labelled_size.py splice-labelled_69 8 1 6-9 --mem 612M --time 1-00:00:00

#poetry run python slurm.py nesi_labelled_size.py anuran-labelled_69 9 1 6-9 --mem 612M --time 1-00:00:00

