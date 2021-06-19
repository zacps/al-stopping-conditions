import argparse
import subprocess
import csv
import io
import requests
import sys

from simple_slurm import Slurm

parser = argparse.ArgumentParser()

parser.add_argument("last_job_id")

parser.add_argument("notebook")
parser.add_argument("job_name")
parser.add_argument("fragment_id", type=int)
parser.add_argument("fragment_length", type=int)
parser.add_argument("fragment_run")
parser.add_argument("--mem", default="612M")
parser.add_argument("--time", default="0-00:02:00")
parser.add_argument("--nobackup", action="store_true")


args = parser.parse_args()

# check if the previous job timed out
out = subprocess.run(
    ["sacct", "-X", "-j", args.last_job_id, "-P"],
    stdout=subprocess.PIPE,
    encoding="utf-8",
)
jobs = csv.DictReader(out.stdout.splitlines(), delimiter="|")

state = next(jobs)["State"]

# if it did schedule the next one with the same settings
if state == "COMPLETED":
    print(f"State was {state}, not scheduling next job")
    sys.exit(0)
if state != "TIMEOUT":
    # TODO: Notify here (discord webhook is the easiest)
    print(f"State was {state}, not scheduling next job")
    requests.post(
        "https://discord.com/api/webhooks/809248326485934080/aIHL726wKxk42YpDI_GtjsqfAWuFplO3QrXoza1r55XRT9-Ao9Rt8sBtexZ-WXSPCtsv",
        data={
            "content": f"Job `{args.last_job_id}` exited with unexpected status `{state}`"
        },
    )
    sys.exit(1)

print("scheduling next job")
command = [
    "poetry",
    "run",
    "python",
    "slurm.py",
    args.notebook,
    args.job_name,
    str(args.fragment_id),
    str(args.fragment_length),
    args.fragment_run,
    f"--mem={args.mem}",
    f"--time={args.time}",
]
if args.nobackup:
    command.append("--nobackup")
subprocess.run(command)
