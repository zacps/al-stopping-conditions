#!/usr/bin/bash
#poetry run python slurm.py nesi_decision_tree.py newsgroups-tree69 0 1 6-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_decision_tree.py rcv1-tree69 1 1 6-9 --mem 4G --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py webkb-tree69 2 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py spamassassin-tree69 3 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py avila-tree69 4 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py smartphone-tree69 5 1 6-9 --mem 612M --time 0-00:02:00 --oneshot
#poetry run python slurm.py nesi_decision_tree.py swarm-tree69 6 1 6-9 --mem 4G --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py sensorless-tree69 7 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py splice-tree69 8 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 
#poetry run python slurm.py nesi_decision_tree.py anuran-tree69 9 1 6-9 --mem 612M --time 0-00:02:00 --oneshot 

poetry run python slurm.py nesi_decision_tree.py newsgroups-tree69 0 1 6-9 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_decision_tree.py rcv1-tree69 1 1 6-9 --mem 4G --time 1-00:00:00 
poetry run python slurm.py nesi_decision_tree.py webkb-tree69 2 1 6-9 --mem 612M --time 1-00:00:00 
poetry run python slurm.py nesi_decision_tree.py spamassassin-tree69 3 1 6-9 --mem 612M --time 0-14:00:00 
poetry run python slurm.py nesi_decision_tree.py avila-tree69 4 1 6-9 --mem 612M --time 0-04:00:00 
poetry run python slurm.py nesi_decision_tree.py smartphone-tree69 5 1 6-9 --mem 612M --time 1-00:00:00
poetry run python slurm.py nesi_decision_tree.py swarm-tree69 6 1 6-9 --mem 4G --time 1-00:00:00 
poetry run python slurm.py nesi_decision_tree.py sensorless-tree69 7 1 6-9 --mem 612M --time 0-09:00:00 
poetry run python slurm.py nesi_decision_tree.py splice-tree69 8 1 6-9 --mem 612M --time 0-04:00:00 
poetry run python slurm.py nesi_decision_tree.py anuran-tree69 9 1 6-9 --mem 612M --time 0-05:00:00 
