#!/bin/bash -l
# Use the exact sbatch command printed by run_explanation_grid --prepare.
#SBATCH --account=aip-zaiane
#SBATCH --job-name=expl_s3r3
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
set -euo pipefail

if [[ $# -ne 1 || -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo 'Use the array command printed by run_explanation_grid --prepare.' >&2
    exit 2
fi
BATCH=$(realpath "$1")
trap 'touch "$BATCH/STOP"' ERR
if [[ ! -f "$BATCH/READY" ]]; then
    echo 'Batch preparation is incomplete.' >&2
    exit 1
fi
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$BATCH/code"
cd "$BATCH/code"
# Python parses CRLF dotenv files as data. Never source the token file.
python -u -m scripts.run_explanation_grid --batch "$BATCH" \
    --run-cell "$SLURM_ARRAY_TASK_ID"
