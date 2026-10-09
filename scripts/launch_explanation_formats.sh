#!/bin/bash
# User-owned: prepare and submit the explanation timing (free) and
# formats (paid) studies from an isolated committed checkout.
#   bash launch_explanation_formats.sh REPO [--suites timing formats]
# Rerunning is safe: a submitted batch is reported, never resubmitted.
set -euo pipefail
repo=$(realpath -- "${1:?Existing repository directory}")
shift
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-explanation-formats-${revision:0:12}"
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
python -m scripts.run_explanation_formats_20260925 --launch \
    --ledger "$repo/results/budget_ledger.json" \
    --credential-file "$repo/.env" "$@"
