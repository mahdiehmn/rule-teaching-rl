#!/bin/bash
# User-owned launcher: bash launch_rule_bank_study.sh <repo> <spec>
set -euo pipefail
repo=$(realpath -- "${1:?Existing repository directory}")
spec=${2:?Study spec, e.g. keycorridor, doorkey_count, doorkey_controls, multiroom_controls, doorkey_online}
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-rule-bank-study-${revision:0:12}"
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
python -m scripts.run_rule_bank_study_20260928 check --spec "$spec"
# Paid studies read the ledger and the key file of the main checkout;
# free studies ignore both.
python -m scripts.run_rule_bank_study_20260928 launch --spec "$spec" \
    --ledger "$repo/results/budget_ledger.json" \
    --credential-file "$repo/.env"
