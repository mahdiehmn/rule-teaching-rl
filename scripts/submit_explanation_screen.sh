#!/bin/bash -l
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
set -euo pipefail
BATCH=$(realpath "$1")
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONPATH="$BATCH/code"
cd "$BATCH/code"
python -u -m scripts.run_explanation_screen \
    --batch "$BATCH" --run-worker "${SLURM_ARRAY_TASK_ID}"
