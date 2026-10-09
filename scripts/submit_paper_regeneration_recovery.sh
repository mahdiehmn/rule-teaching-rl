#!/bin/bash -l
# User-submitted worker; execute only the recovery batch's frozen source.
#SBATCH --account=aip-zaiane
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --no-requeue
set -euo pipefail
batch=$(realpath -- "${1:?Recovery batch directory}")
stage=${2:?recover or train}
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$batch/code"
cd "$batch/code"
case "$stage" in
    recover)
        exec python -u -m scripts.recover_paper_regeneration_20261005 \
            recover --batch "$batch"
        ;;
    train)
        exec python -u -m scripts.recover_paper_regeneration_20261005 \
            train --batch "$batch" --index "${SLURM_ARRAY_TASK_ID:?Array index}"
        ;;
    *)
        printf 'Unknown recovery stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac
