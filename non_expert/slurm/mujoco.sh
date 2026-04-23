#!/bin/bash
#SBATCH --job-name=mujoco-sweep
#SBATCH --partition=a100
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=logs/mujoco_%A_%a.out
#SBATCH --error=logs/mujoco_%A_%a.err

set -euo pipefail

#LAMBDAS=(1 2 5 10, 20, 50, 100, 1000)
#SIGMAS=(0.5 1.0 2.0, 3.0)

LAMBDAS=(10, 20, 50, 100)
SIGMAS=(2.0, 3.0, 5.0)

NUM_SIGMAS=${#SIGMAS[@]}

LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SIGMAS))
SIGMA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SIGMAS))

LAMBD=${LAMBDAS[$LAMBDA_INDEX]}
SIGMA=${SIGMAS[$SIGMA_INDEX]}

echo "Running lambda=${LAMBD}, sigma=${SIGMA}"
echo "Array task id: ${SLURM_ARRAY_TASK_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"

mkdir -p logs

# Activate your environment here, for example:
# source ~/.bashrc
ml anaconda
conda activate npmpc
# or:
# source /path/to/venv/bin/activate
cd /home/acaste11/scr4_emallad1/agu/npmpc

python -m non_expert.main_mujoco \
  --lambd "${LAMBD}" \
  --sigma "${SIGMA}" \
  --faiss-threads "${SLURM_CPUS_PER_TASK}"
  --env-id "Pendulum-v1"

