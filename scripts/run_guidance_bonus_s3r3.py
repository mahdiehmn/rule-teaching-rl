"""
Run one predeclared cell; --dry-run prints its full configuration.

Indices 0..4: no guidance/no bonus; 5..9: guidance/no bonus;
10..14: no guidance/count; 15..19: guidance/count. Each block uses
training seeds 0..4. All arms execute student actions and evaluate
without a teacher. This is a bot action-target experiment, not an
LLM-text or rationale experiment.
"""

import argparse
from dataclasses import asdict
import json

from algos.ppo_distill import Args, train


EXPERIMENT_ID = 'guidance_bonus_s3r3_v1'


def config(index):
    """
    Build matched configurations without inheriting trainer drift.
    """

    if index not in range(20):
        raise ValueError('index must be in 0..19')
    arm, seed = divmod(index, 5)
    return Args(
        experiment_id=EXPERIMENT_ID,
        task='keycorridor_s3r3', seed=seed,
        obs_mode='symbolic', agent_view_size=7, obs_target_size=56,
        recurrent=True, dual_value=True, cuda=False,
        guidance=bool(arm % 2), bonus='count' if arm >= 2 else 'none',
        teacher='bot', advisor='unlimited', query_interval=1,
        advice_replay=False, advice_budget=0, query_budget=0,
        total_timesteps=10_000_000, learning_rate=2.5e-4,
        num_envs=8, num_steps=128, num_minibatches=4, update_epochs=4,
        gamma=0.999, int_gamma=0.99, gae_lambda=0.95,
        int_coef=1.0, norm_int_reward=True, coverage_log=True,
        ent_coef=0.01, vf_coef=0.5, clip_coef=0.2,
        anneal_lr=True, norm_adv=True, clip_vloss=True,
        max_grad_norm=0.5, target_kl=None,
        distill_coef_start=10.0, distill_coef_min=0.1,
        distill_fraction=0.5, distill_cutoff=0.75,
        distill_delay_frac=0.0,
        eval_interval=200, eval_episodes=50, eval_sampled=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=range(20), required=True)
    parser.add_argument('--dry-run', action='store_true')
    cli = parser.parse_args()
    args = config(cli.index)
    print(json.dumps(asdict(args), indent=2), flush=True)
    if not cli.dry_run:
        train(args)


if __name__ == '__main__':
    main()
