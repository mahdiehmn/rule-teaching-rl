#!/bin/bash -l
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --no-requeue
set -euo pipefail
batch=$(realpath -- "${1:?Batch directory}")
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$batch/code"
cd "$batch/code"
exec python -u -m scripts.run_rule_bank_pilot_20260928 run-cell \
    --batch "$batch" --index "${SLURM_ARRAY_TASK_ID:?Array index}"
