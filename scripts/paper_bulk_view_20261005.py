"""Fresh teacher-view replication for the bulk batch.

Both arms keep the student's local observation
and the same frozen rules. Only the rule executor's observer changes.
"""

from dataclasses import replace
from pathlib import Path

from scripts import run_fix_wave_20260929 as fw
from scripts import teacher_view_20261001 as tv

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_bulk_view_20261005_v1'
PROTOCOL = 'research/paper_bulk_view_2026-10-05.md'
TIME = '1-06:00:00'
REPS = range(100, 110)
ARMS = ('local', 'full_map')
BACKGROUNDS = ('none', 'count')
SOURCE_ARMS = (
    ('dk_fresh', 'rules_weak'),
    ('mr_fresh', 'rules_weak'),
    ('kc_fresh', 'rules_mem_weak'),
)
PRIMARY_CONTRASTS = tuple(
    (fw.SUITES[suite][0], bonus, 'full_map', 'local')
    for suite, _arm in SOURCE_ARMS for bonus in BACKGROUNDS
)
REQUIRED = (
    PROTOCOL, 'scripts/paper_bulk_view_20261005.py',
    'scripts/teacher_view_20261001.py',
    'teachers/minigrid/teacher_view.py',
    'teachers/minigrid/rule_bank.py',
    *tv.BANKS, *tv.BANKS.values(),
)


def cells(root=ROOT):
    """
    Construct ten fresh paired replicates for each of six comparisons.
    """

    # Reject drift in any copied bank before creating runnable cells.
    # The provenance field is descriptive; rules must match exactly.
    tv.check_banks(root)
    result = []
    for replicate in REPS:
        for source_suite, source_arm in SOURCE_ARMS:
            for bonus in BACKGROUNDS:
                for arm in ARMS:
                    bank_arm = (source_arm if arm == 'local'
                                else source_arm + '_tv')
                    args = fw.arm_args(
                        source_suite, bank_arm, replicate, bonus, root)
                    args = replace(
                        args,
                        experiment_id=(
                            f'{STUDY}_{args.task}_{bonus}_{arm}'),
                        diagnostic_eval_frames=fw.DENSE_FRAMES,
                        rule_timing_diagnostics=True)
                    fw.cell(result, STUDY, arm, args, replicate)
    return result
