#!/bin/bash -l
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=4-00:00:00
#SBATCH --no-requeue
# One writer job (collect, freeze, gate, calibrate) or one learning cell,
# always from the batch's archived code. Submitted only by the runner.
set -euo pipefail
batch=$(realpath -- "${1:?Batch directory}")
stage=${2:?writer or cell}
target=${3:?writer access or learning suite}
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$batch/code"
cd "$batch/code"
if [[ "$stage" == writer ]]; then
    exec python -u -m scripts.run_explanation_formats_20260925 \
        --batch "$batch" --writer "$target"
fi
test "$stage" = cell
exec python -u -m scripts.run_explanation_formats_20260925 \
    --batch "$batch" --suite "$target" \
    --run-cell "${SLURM_ARRAY_TASK_ID:?Array index}"
