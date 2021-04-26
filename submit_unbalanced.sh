#!/bin/bash
#poetry run python slurm.py nesi_unbalanced.py avila-unbalanced 0 1 0-1 --mem 612M --time 0-00:02:00
poetry run python slurm.py nesi_unbalanced.py sensorless-unbalanced 1 1 0-1 --mem 612M --time 1-00:00:00
#poetry run python slurm.py nesi_unbalanced.py anuran-unbalanced 2 1 0-1 --mem 612M --time 0-00:02:00

