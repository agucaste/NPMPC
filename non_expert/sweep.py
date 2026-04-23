# non_expert/sweep_mujoco.py
import itertools
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

LAMBDAS = [5, 10, 20]
SIGMAS = [0.5, 2.0, 3.0]
MAX_WORKERS = 3


def run_one(params):
    lambd, sigma = params

    cmd = [
        sys.executable,
        "-m",
        "non_expert.main_mujoco",
        "--lambd",
        str(lambd),
        "--sigma",
        str(sigma),
    ]

    print(f"Starting lambda={lambd}, sigma={sigma}")
    result = subprocess.run(cmd)
    return lambd, sigma, result.returncode


if __name__ == "__main__":
    jobs = list(itertools.product(LAMBDAS, SIGMAS))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_one, job) for job in jobs]

        for future in as_completed(futures):
            lambd, sigma, returncode = future.result()
            print(f"Finished lambda={lambd}, sigma={sigma}, returncode={returncode}")
