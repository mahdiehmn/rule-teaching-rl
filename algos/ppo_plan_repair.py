"""
Prospective same-input plan repair diagnostic with no online teacher.

The inherited adapter supplies exactly the original
plan architecture, replay selection and loss. This wrapper only validates the
new arm/source contracts and records both raw and corrected audit metrics.
"""

from dataclasses import dataclass
import json

import torch
import tyro

from algos import ppo_distill as ppo
from algos import ppo_lesson_formats as formats
from scripts.explanation_screen_panel import file_hash
from scripts.plan_repair_targets import (
    EXPECTED_COUNTS, VERSION, calibration_scale, read_json,
    validate_derived_bank, validate_sources)

ARMS = ('ppo', 'raw', 'corrected', 'permuted_corrected', 'detached')


@dataclass
class Args(formats.Args):
    repair_arm: str = 'raw'
    raw_bank: str = ''
    raw_bank_sha256: str = ''
    repair_panel: str = ''
    repair_panel_sha256: str = ''
    repair_calibration: str = ''
    repair_calibration_sha256: str = ''


def validate_args(args, expected_counts=EXPECTED_COUNTS):
    """
    Reject factor drift and verify every source before the trainer uses it.
    """

    if (args.repair_arm not in ARMS or args.task != 'doorkey_8x8'
            or args.bonus != 'count' or args.obs_mode != 'symbolic'
            or not args.recurrent or not args.dual_value or args.cuda
            or args.guidance or args.query_budget != 0
            or args.lesson_action_coef != 0 or args.lesson_format != 'plan'
            or args.format_access != 'full_state'
            or args.explanation != 'none' or args.consequence != 'none'
            or args.advice_replay):
        raise ValueError('Unsupported Count-DoorKey plan-repair configuration')
    mode = ('ppo' if args.repair_arm == 'ppo' else 'detached'
            if args.repair_arm == 'detached' else 'explanation')
    target = ('permuted' if args.repair_arm == 'permuted_corrected'
              else 'aligned')
    if args.lesson_mode != mode or args.lesson_targets != target:
        raise ValueError('Repair arm disagrees with lesson mode or targets')
    for path, expected in (
            (args.raw_bank, args.raw_bank_sha256),
            (args.repair_panel, args.repair_panel_sha256),
            (args.repair_calibration, args.repair_calibration_sha256),
            (args.format_bank, args.format_bank_sha256)):
        if file_hash(path) != expected:
            raise ValueError('Frozen plan-repair source hash differs')
    raw, panel, calibration = (read_json(path) for path in (
        args.raw_bank, args.repair_panel, args.repair_calibration))
    scale = calibration_scale(calibration, args.raw_bank_sha256, raw)
    if args.lesson_scale != (0.0 if mode == 'ppo' else scale):
        raise ValueError('Every active arm must reuse the raw plan scale')
    bank = read_json(args.format_bank)
    if args.repair_arm in ('ppo', 'raw'):
        if (args.format_bank_sha256 != args.raw_bank_sha256 or bank != raw):
            raise ValueError('PPO and raw arms must use the original bank')
        validate_sources(raw, panel, expected_counts)
    else:
        validate_derived_bank(bank, raw, panel, args.raw_bank_sha256,
                              args.repair_panel_sha256, expected_counts)
    return raw


class PlanRepairAuxiliary(formats.FormatLessonAuxiliary):
    """
    Apply the unchanged plan objective after the prospective bank checks.
    """

    def initialize(self, agent, run_dir, device, args):
        self.raw_bank = validate_args(args)
        super().initialize(agent, run_dir, device, args)
        path = self.directory / 'lesson_contract.json'
        contract = read_json(path)
        contract['plan_repair'] = dict(
            version=VERSION, arm=args.repair_arm,
            raw_bank_sha256=args.raw_bank_sha256,
            panel_sha256=args.repair_panel_sha256,
            calibration_sha256=args.repair_calibration_sha256,
            scale_source='Original raw bank primary plan scale; no refit',
            no_new_teacher_requests=True,
            native_corrected_truth_checked=True,
            old_row_checks='Historical raw-reply checks, unchanged')
        path.write_text(json.dumps(contract, indent=2) + '\n',
                        encoding='utf-8')

    def finish(self, global_step):
        super().finish(global_step)
        path = self.directory / 'lesson_finished.json'
        finished = read_json(path)
        finished['plan_repair'] = dict(
            version=VERSION, arm=self.args.repair_arm,
            raw_bank_sha256=self.args.raw_bank_sha256,
            panel_sha256=self.args.repair_panel_sha256,
            calibration_sha256=self.args.repair_calibration_sha256)
        finished['semantic_correctness'] = (
            'Corrected categorical plans crosschecked against native state; '
            'raw prose and other formats unchanged and not certified')
        if self.raw_bank['audit']:
            rows = self.raw_bank['audit']
            with torch.no_grad():
                obs = torch.tensor([r['observation'] for r in rows],
                                   device=self.device).float()
                pos = torch.tensor([r['positive_action'] for r in rows],
                                   device=self.device)
                neg = torch.tensor([r['foil_action'] for r in rows],
                                   device=self.device)
                x = formats.head_input(self.agent._encode(obs), pos, neg)
                scores = {}
                for source in ('raw', 'native_corrected'):
                    targets = [r['targets'] if source == 'raw' else
                               {'plan': r['checks']['plan_truth']}
                               for r in rows]
                    target = formats.target_tensors(targets, 'plan',
                                                    self.device)
                    matches = {k: head(x).argmax(-1).eq(target[k])
                               for k, head in self.heads.items()}
                    scores[source] = {k: float(v.float().mean())
                                      for k, v in matches.items()}
                    scores[source]['exact_bundle'] = float(torch.stack(
                        list(matches.values())).all(0).float().mean())
                finished['audit_plan_metrics'] = scores
                finished['audit_plan_head_trained'] = self.uses_explanation
        path.write_text(json.dumps(finished, indent=2) + '\n',
                        encoding='utf-8')


if __name__ == '__main__':
    ppo.train(tyro.cli(Args), auxiliary=PlanRepairAuxiliary())
