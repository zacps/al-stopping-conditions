import argparse

from simple_slurm import Slurm

parser = argparse.ArgumentParser()

parser.add_argument('job_name')
parser.add_argument('fragment_id', type=int)
parser.add_argument('fragment_length', type=int)
parser.add_argument('fragment_run')

args = parser.parse_args()


slurm = Slurm(
    #array=range(0,1),
    cpus_per_task=2,
    job_name=args.job_name,
    time="1-00:00:00",
    #time="0-01:00:00",
    mem_per_cpu="512MB",
    ntasks=1,
    output=f"/nesi/project/uoa03271/logs/{Slurm.JOB_ID}_{Slurm.JOB_NAME}_{Slurm.JOB_ARRAY_ID}.txt"
)

slurm.sbatch(f"""
module load Python/3.8.2-gimkl-2020a
poetry run python nesi.py {args.fragment_id} {args.fragment_length} {args.fragment_run}
""")
