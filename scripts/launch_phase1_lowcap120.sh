#!/bin/bash
# Run by the user; keep the cluster checkout and old jobs intact.
# Usage: ... | bash -s -- <repo> [doorkey] [multiroom] [keycorridor]
# Prepares the named batches (all three by default; one hold each) and
# submits one array per task. Later arrays in the same call start 20 and
# 40 minutes later to spread the early entropy-paced GPT requests.
#
# Optional environment, for clusters other than Vulcan (e.g. Fir):
#   DATA=<dir>            holds advising_strength/<reused batches>
#                         (default: <repo>/results)
#   ACCOUNT=<account>     Slurm account override, e.g. def-zaiane_cpu
#   INIT_LEDGER_USD=<x>   create results/budget_ledger.json with this
#                         allowance first; refuses if a ledger exists
set -euo pipefail
repo=$(realpath -- "${1:?Pass the existing repository directory}")
shift
suites=("$@")
if (( ${#suites[@]} == 0 )); then
    suites=(doorkey multiroom keycorridor)
fi
for suite in "${suites[@]}"; do
    case "$suite" in
        doorkey|multiroom|keycorridor) ;;
        *) echo "Unknown task: $suite" >&2; exit 2 ;;
    esac
done
DATA=${DATA:-}
ACCOUNT=${ACCOUNT:-}
INIT_LEDGER_USD=${INIT_LEDGER_USD:-}
data_args=()
if [[ -n "$DATA" ]]; then
    DATA=$(realpath -- "$DATA")
    test -d "$DATA/advising_strength"
    data_args=(--data "$DATA")
fi
test "$(git -C "$repo" rev-parse --show-toplevel)" = "$repo"
revision=$(git -C "$repo" rev-parse --verify 'FETCH_HEAD^{commit}')
launch="$(dirname -- "$repo")/.vlm-phase1-lowcap-${revision:0:12}"
if [[ -e "$launch" ]]; then
    test "$(git -C "$launch" rev-parse HEAD)" = "$revision"
    git -C "$launch" diff --quiet HEAD --
else
    git -C "$repo" worktree add --detach "$launch" "$revision"
fi
mkdir -p "$repo/results"
for name in results .env; do
    if [[ -e "$launch/$name" || -L "$launch/$name" ]]; then
        test "$(realpath -- "$launch/$name")" = "$(realpath -- "$repo/$name")"
    else
        ln -s -- "$repo/$name" "$launch/$name"
    fi
done
module load python/3.11
source "${VLM_BENCH_ENV:-$HOME/vlmbench-env}/bin/activate"
cd "$launch"
if [[ -n "$INIT_LEDGER_USD" ]]; then
    python - "$INIT_LEDGER_USD" <<'PY'
import sys
from pathlib import Path
from teachers.budget import CostLedger
path = Path('results/budget_ledger.json')
if path.exists():
    raise SystemExit(f'{path} already exists; not re-initializing')
CostLedger.initialize(str(path), float(sys.argv[1]),
                      note='Separate cluster ledger for phase1_lowcap120 batches')
print('Initialized', path.resolve(), 'allowance', sys.argv[1])
PY
fi
for suite in "${suites[@]}"; do
    python -m scripts.run_phase1_lowcap120 --prepare --suite "$suite" \
        ${data_args[@]+"${data_args[@]}"}
done
DATA="$DATA" ACCOUNT="$ACCOUNT" python - "${suites[@]}" <<'PY'
import os
from pathlib import Path
import shlex
import subprocess
import sys
from scripts import run_phase1_lowcap120 as study

root = Path.cwd()
data = Path(os.environ['DATA']) if os.environ.get('DATA') else None
ledger = root / 'results/budget_ledger.json'
jobs = {'doorkey': 'p1lc_dk', 'multiroom': 'p1lc_mr', 'keycorridor': 'p1lc_kc'}
for order, suite in enumerate(sys.argv[1:]):
    job = jobs[suite]
    begin = f'now+{20 * order}minutes' if order else None
    manifest = study.make_manifest(suite, root, data)
    batch = study.transfer.existing_batch(root, ledger, manifest)
    if batch is None or (batch / 'STOP').exists():
        raise RuntimeError('Batch missing or stopped; inspect before submitting')
    if (batch / 'SUBMITTED_JOB').exists():
        print(job, 'already submitted:', (batch / 'SUBMITTED_JOB').read_text().strip())
        continue
    # Measured: these runs take <= 8.8 h and <= 0.92 GB. The worker's own
    # 4-day/20 GB defaults route them to the long, scarce CPU partition.
    command = ['sbatch', '--parsable', '--job-name=' + job,
               '--array=0-' + str(len(manifest['cells']) - 1),
               '--time=1-00:00:00', '--mem=4G',
               f'--output={batch}/slurm/%x_%A_%a.out']
    if os.environ.get('ACCOUNT'):
        command.append('--account=' + os.environ['ACCOUNT'])
    if begin:
        command.append('--begin=' + begin)
    command += [str(batch / 'code' / study.paid.WORKER), str(batch)]
    # Exclusive marker protects against accidental repeated submissions.
    with (batch / 'SUBMISSION_ATTEMPTED').open('x') as handle:
        handle.write(shlex.join(command) + '\n')
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    (batch / 'SUBMITTED_JOB').write_text(result.stdout)
    print(job + ':', result.stdout.strip())
PY
