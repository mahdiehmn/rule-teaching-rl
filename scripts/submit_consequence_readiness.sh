#!/bin/bash -l
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:15:00
#SBATCH --no-requeue
set -euo pipefail
batch=$(realpath "$1")
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2
export PYTHONPATH="$batch/code"
cd "$batch/code"
python -u -m scripts.run_consequence_readiness --batch "$batch" --run-worker "${SLURM_ARRAY_TASK_ID}"
