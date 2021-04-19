import argparse

from simple_slurm import Slurm

parser = argparse.ArgumentParser()

parser.add_argument('notebook')
parser.add_argument('job_name')
parser.add_argument('fragment_id', type=int)
parser.add_argument('fragment_length', type=int)
parser.add_argument('fragment_run')
parser.add_argument('--mem', default='612M')
parser.add_argument('--time', default='0-00:02:00')

args = parser.parse_args()


slurm = Slurm(
    #array=range(0,1),
    cpus_per_task=2,
    job_name=args.job_name,
    time=args.time,
    mem_per_cpu=args.mem,
    #ntasks=1,
    output=f"/nesi/project/uoa03271/logs/{Slurm.JOB_NAME}_{Slurm.JOB_ID}_{Slurm.JOB_ARRAY_ID}.txt"
)

job_id = slurm.sbatch(f"""
module load Python/3.8.2-gimkl-2020a
poetry run python checker.py ${SLURM_JOB_ID}
poetry run python {notebook} {args.fragment_id} {args.fragment_length} {args.fragment_run}
""")

# ---------------------------------------------------------------------------------------------------

slurm = Slurm(
    cpus_per_task=2,
    job_name=f"{args.job_name}-check",
    time="0-00:01:00",
    mem_per_cpu="10M",
    afternotok=job_id,
    output=f"/nesi/project/uoa03271/logs/{Slurm.JOB_NAME}_{Slurm.JOB_ID}_{Slurm.JOB_ARRAY_ID}.txt"
)

# checker.py will check to see if the above job exited with timeout, and if it did schedule an identical jobs by calling this again.
# then, this will happen recursively until there's a non-timeout exit code.
slurm.sbatch(f"""
poetry run python checker.py {job_id} {notebook} {args.job_name} {args.fragment_id} {args.fragment_length} {args.fragment_run} {args.mem} {args.time}
""")
