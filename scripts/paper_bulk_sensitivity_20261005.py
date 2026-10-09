"""Frozen KeyCorridor weight and advice-duration grid for the bulk wave.

No outcome selects a cell in this grid.
"""

from dataclasses import replace
from pathlib import Path

from scripts import run_fix_wave_20260929 as fw

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper_bulk_sensitivity_20261005_v1'
PROTOCOL = 'research/paper_bulk_protocol_2026-10-05.md'
TIME = '3-00:00:00'
HORIZON = 30_000_000
SPECS = {
    'kc_s4_weights': ('kc_s4_long', range(110, 120)),
    'kc_s6_weights': ('kc_s6_long', range(120, 130)),
}
WEIGHTS = {'weight_001': .01, 'weight_01': .1, 'weight_1': 1.0,
           'weight_01_fixed15m': .1}
ARMS = ('none', *WEIGHTS)
REQUIRED = ('scripts/paper_bulk_sensitivity_20261005.py', PROTOCOL)


def cells(root=ROOT, suite=None):
    """
    Build ten paired seeds per map/student/arm at one common horizon.
    """

    selected = SPECS if suite is None else {suite: SPECS[suite]}
    result = []
    for name, (source, reps) in selected.items():
        for replicate in reps:
            for bonus in ('none', 'count'):
                for arm in ARMS:
                    base_arm = 'none' if arm == 'none' else 'rules_mem_weak'
                    args = fw.arm_args(source, base_arm, replicate,
                                       bonus, root)
                    args = replace(
                        args, total_timesteps=HORIZON,
                        experiment_id=f'{STUDY}_{name}_{bonus}_{arm}',
                        diagnostic_eval_frames=fw.DENSE_FRAMES,
                        eval_frame_milestones=(
                            '2000000,5000000,15000000,30000000'),
                        rule_timing_diagnostics=arm != 'none')
                    if arm != 'none':
                        args = replace(args, distill_coef_start=WEIGHTS[arm],
                                       distill_coef_min=WEIGHTS[arm] / 100)
                    if arm == 'weight_01_fixed15m':
                        # The 30M run has twice as many rollouts
                        # as 15M. Halving both fractions preserves the
                        # old advice schedule in rollout units.
                        args = replace(args, distill_fraction=.25,
                                       distill_cutoff=.375)
                    fw.cell(result, name, arm, args, replicate)
    return result


def families():
    """
    Separate declared weight/net-benefit and withdrawal comparisons.
    """

    tasks = ('keycorridor_s4r3', 'keycorridor_s6r3_babyai')
    return {
        'weight_and_net_benefit': [
            (task, bonus, arm, 'none') for task in tasks
            for bonus in ('none', 'count') for arm in WEIGHTS],
        'advice_duration': [
            (task, bonus, 'weight_01_fixed15m', 'weight_01')
            for task in tasks for bonus in ('none', 'count')],
    }
