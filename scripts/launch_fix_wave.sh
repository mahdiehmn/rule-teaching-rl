#!/bin/bash
# User-owned launcher: bash launch_fix_wave.sh <repo> [suite ...]
# Suites default to all fourteen; each is its own array. Re-running is safe:
# a suite already submitted is checked and reported, never resubmitted.
set -euo pipefail
repo=$(realpath -- "${1:?Existing repository directory}")
shift
if [[ $# -gt 0 ]]; then suites=("$@")
else suites=(kc mr dk kc_long kc_llm mr_llm dk_llm
                   kc_confirm mr_confirm dk_confirm kc_mem kc_mem_confirm dk16 kc_s4); fi
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-fix-wave-${revision:0:12}"
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
for suite in "${suites[@]}"; do
    python -m scripts.run_fix_wave_20260929 check --suite "$suite"
    python -m scripts.run_fix_wave_20260929 launch --suite "$suite"
done
