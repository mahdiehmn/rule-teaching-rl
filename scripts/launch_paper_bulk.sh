#!/bin/bash
# User-owned entry: bash launch_paper_bulk.sh <repo> [group ...]
# With no group arguments, admit and queue the entire 640-run package.
set -euo pipefail
repo=$(realpath -- "${1:?Existing repository directory}")
shift
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-paper-bulk-${revision:0:12}"
if [[ -e "$launch" ]]; then
    test "$(git -C "$launch" rev-parse HEAD)" = "$revision"
    git -C "$launch" diff --quiet HEAD --
else
    git -C "$repo" worktree add --detach "$launch" "$revision"
fi
if [[ -e "$launch/results" || -L "$launch/results" ]]; then
    test "$(realpath -- "$launch/results")" = "$(realpath -- "$repo/results")"
else
    mkdir -p -- "$repo/results"
    ln -s -- "$repo/results" "$launch/results"
fi
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$launch"
groups=()
if [[ $# -gt 0 ]]; then groups=(--groups "$@"); fi
python -m scripts.run_paper_bulk_20261005 launch "${groups[@]}"
printf 'Bulk launch finished. Worktree: %s\n' "$launch"
