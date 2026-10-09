#!/bin/bash -l
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --no-requeue
# One Crafter run of the frozen manifest (scripts/run_crafter_vulcan_20261002.py).
set -euo pipefail
code=$(realpath -- "${1:?Launch worktree}")
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$code"
cd "$code"
exec python -u -m scripts.run_crafter_vulcan_20261002 run-cell \
    --index "${SLURM_ARRAY_TASK_ID:?Array index}"
