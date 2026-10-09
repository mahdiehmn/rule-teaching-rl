#!/bin/bash
# User-owned launcher for the Crafter wave on Vulcan:
#   bash scripts/launch_crafter_vulcan.sh <repo>
# Runs from a detached worktree at FETCH_HEAD (as launch_fix_wave.sh does),
# so later commits cannot change a live run. Safe to re-run: an already
# submitted wave is reported, never resubmitted.
set -euo pipefail
repo=$(realpath -- "${1:?Existing repository directory}")
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-crafter-${revision:0:12}"
if [[ -e "$launch" ]]; then
    test "$(git -C "$launch" rev-parse HEAD)" = "$revision"
    git -C "$launch" diff --quiet HEAD --
else
    git -C "$repo" worktree add --detach "$launch" "$revision"
fi
if [[ -e "$launch/results" || -L "$launch/results" ]]; then
    test "$(realpath -- "$launch/results")" = "$(realpath -- "$repo/results")"
else
    ln -s -- "$repo/results" "$launch/results"
fi
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
cd "$launch"
python -c "import crafter" || {
    echo "crafter is not installed in this environment:"
    echo "  pip install crafter==1.8.3"
    exit 1
}
python -m scripts.run_crafter_vulcan_20261002 check
python -m scripts.run_crafter_vulcan_20261002 prepare
batch="$launch/results/crafter_vulcan/crafter_vulcan_20261002"
if [[ -e "$batch/SUBMITTED_JOB" ]]; then
    echo "crafter wave already submitted: $(cat "$batch/SUBMITTED_JOB")"
    exit 0
fi
mkdir -p "$batch/slurm"
job=$(sbatch --parsable --job-name=crafter_wave --array=0-49 \
    --time=12:00:00 --output="$batch/slurm/%x_%A_%a.out" \
    scripts/submit_crafter_vulcan.sh "$launch")
echo "$job" > "$batch/SUBMITTED_JOB"
echo "crafter wave: $job (50 cells)"
