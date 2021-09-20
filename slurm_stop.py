import argparse
import sys

from simple_slurm import Slurm

parser = argparse.ArgumentParser()

parser.add_argument("job_name")
parser.add_argument("fragment_id", type=int)
parser.add_argument("fragment_length", type=int)
parser.add_argument("fragment_run")
parser.add_argument("--mem", default="612M")
parser.add_argument("--jobs", default="20")
parser.add_argument("--time", default="0-02:00:00")

args = parser.parse_args()

fragment_run = args.fragment_run.split("-")
start = int(fragment_run[0])
if len(fragment_run) == 2:
    end = int(fragment_run[1])
else:
    end = None

slurm = Slurm(
    cpus_per_task=args.jobs,
    job_name=args.job_name,
    time=args.time,
    mem_per_cpu=args.mem,
    output=f"/nesi/project/uoa03271/logs/{Slurm.JOB_NAME}_{Slurm.JOB_ID}_{Slurm.JOB_ARRAY_ID}.txt",
)

job_id = slurm.sbatch(
    f"""
module load Python/3.8.2-gimkl-2020a
poetry run python stop_eval.py {args.fragment_id} {args.fragment_length} {args.fragment_run} --jobs={args.jobs}
"""
)
