"""
PPO with annealed teacher distillation (kickstarting).

The update-level integration channel: the student ALWAYS samples
and executes its own action -- the rollout is byte-for-byte the
vanilla PPO rollout, so it stays genuinely on-policy -- and the
teacher's advice enters only as an auxiliary cross-entropy term in
the loss, annealed away over training:

    L = L_PPO + lambda(t) * CE( q_teacher(.|s) || pi_theta(.|s) )

evaluated on the states the student itself visited. This is
Kickstarting (Schmitt et al. 2018) and the mechanism of LLM4Teach
(Zhou et al., IJCAI 2024), whose released code was verified
directly: student acts itself, teacher distribution stored beside
the rollout, coefficient decayed to a floor then hard-dropped.

Contrast with algos/ppo_teacher.py (action override): there the
teacher's action replaces the student's in the environment, which
breaks PPO's on-policy ratio and collapsed on teacher-off handoff
(docs/progress_june_slides.md). Here nothing the teacher says ever
reaches the environment, the critic only ever sees the student's
own trajectories, and the final `1 - distill_cutoff` fraction of
every run is pure PPO -- so teacher-off retention is measured
within every single run.

Teachers (see teachers/factory.py and docs/stage1_distill.md):
  --teacher oracle
      BFS DoorKey oracle (doorkey_* tasks). Free, offline,
      deterministic.
  --teacher bot
      BabyAI Bot (any BabyAI-registered task: gotoseq*,
      keycorridor_s3r3, keycorridor_s6r3_babyai). Free, offline,
      deterministic.
  --teacher vlm_general
      OpenAI VISION model reasoning over the rendered map image;
      any task. Costs money, so submit these deliberately rather
      than folding them into a sweep -- but they DO run via
      scripts/submit_cc.sh. Compute nodes have outbound internet
      (verified 2026-08-27: api.openai.com answers 401 from a
      compute node, i.e. the request arrives).
  --teacher llm_general
      OpenAI TEXT-ONLY model reasoning over an ASCII map + object
      list; any task. Same cost/internet caveats as vlm_general.
      Comparing llm_general vs vlm_general on one task isolates
      perception from planning (the DoorKey finding: VLMs fail at
      reading the image, not at planning).

For the paid teachers, --query-interval k caps API cost by only
querying the teacher every k-th rollout step (the distillation
mask already handles unqueried steps); cost scales ~1/k.

Run it locally:

    python -m algos.ppo_distill --task doorkey_8x8 --teacher oracle
    python -m algos.ppo_distill --task gotoseq --teacher bot
    python -m algos.ppo_distill --task keycorridor_s6r3 \\
        --teacher vlm_general --query-interval 8
"""

import hashlib
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from algos.explanation_head import (
    PhaseShuffler,
    explanation_loss,
)
from envs.phases import s3r3_phase
from teachers.embeddings import EmbeddingProvider
from algos.nets import ObsEncoder, RecurrentCore, is_symbolic_mode
from algos.distill_bonus import (
    CountExploration, intrinsic_gae, local_count_observations,
)
from monitoring.evaluation_diagnostics import (
    diagnostic_milestones, isolated_evaluation_rng, policy_sha256,
)
from envs.registry import TASKS, build_env, make_thunk
from advising import PeekableTeacher, make_advisor
from advising.importance import make_teacher_q_importance
from advising.schedule import QueryWindows
from teachers.budget import (
    BudgetError, CostLedger, PriceTable, reservation_for,
)
from advising.policy import (
    logit_gap_importance,
    max_prob_importance,
    normalized_entropy_importance,
    policy_entropy_importance,
    probability_mistake,
    top2_gap_importance,
)
from advising.replay import AdviceReplay
from envs.state import extract_doorkey_state
from monitoring.eval import greedy_eval
from monitoring.coverage import TrainingCoverage
from monitoring.consultations import ConsultationJournal
from monitoring.metrics import RunTracker, runtime_metadata
from teachers.base import Cost
from teachers.factory import make_teacher


@dataclass
class Args:
    """
    Command-line arguments for a distillation run.

    PPO knobs match algos/ppo.py exactly (same defaults, same
    meanings) so distill-vs-baseline differences are attributable
    to the teacher term, not to tuning drift. New knobs are the
    teacher block and the distillation schedule at the bottom.
    """

    # --- Experiment identity and logging ---
    task: str = 'gotoseq'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = 'vlm-rl-bench'
    experiment_id: str = ''

    # --- Observation settings ---
    # 'historical' (fog of war) is the project's standard for all
    # Stage 1+ comparisons -- see docs/stage0_ppo.md for why it is
    # the mode that keeps "partial" consistent across map sizes.
    obs_mode: str = 'historical'
    agent_view_size: int = 7
    obs_tile_size: int = 0
    obs_target_size: int = 56
    # Add a GRU core between the trunk and the heads. Needed to run a
    # teacher arm at the SAME configuration as the exploration-bonus
    # arms: on keycorridor_s3r3 those reach 1.00 at symbolic+GRU and
    # 0.00 on RGB, so a teacher measured only on RGB differs from
    # them in three variables at once and licenses no comparison.
    recurrent: bool = False

    # The factorial comparison uses the same optional second critic
    # in all four arms. Legacy checkpoints keep their original shape.
    guidance: bool = True
    bonus: str = 'none'
    dual_value: bool = False
    int_gamma: float = 0.99
    int_coef: float = 1.0
    norm_int_reward: bool = True
    coverage_log: bool = False
    # Redirect a share of labels as a fixed function of hidden state,
    # to test whether label ambiguity alone can harm a competent
    # student. Zero reproduces every previous experiment exactly.
    advice_alias_rate: float = 0.0
    # Deliver a usable label only if the evidence the teacher's own
    # reason names is inside the student's current view ('visible'),
    # a same-rate random placebo ('random') or only the unseen ones
    # ('inverse'). 'none' reproduces every previous experiment exactly.
    # See teachers/evidence.py.
    advice_evidence_gate: str = 'none'
    # Dose: deliver at most round(f * usable) labels per rollout, drawn
    # uniformly from the gate's pool (see EvidenceGate.finalize). 1.0
    # reproduces every previous experiment exactly.
    advice_keep_fraction: float = 1.0
    # Read-only applicability/exposure logs for the timing experiment.
    rule_timing_diagnostics: bool = False
    # ADVISOR (Weihs et al., NeurIPS 2021): an auxiliary actor on the
    # shared features, trained by imitation only, estimates the
    # imitation gap per state; w = exp(-alpha * CE(teacher, aux)) weights
    # imitation and (1 - w) the policy-gradient loss. 'none' reproduces
    # every previous experiment exactly.
    imitation_weighting: str = 'none'
    advisor_alpha: float = 4.0

    # Separate policy representation from the count reward's input.
    # 'policy' preserves all historical experiments. The new control
    # hashes the native local symbolic view for either policy view.
    count_observation: str = 'policy'

    # --- Core PPO hyperparameters (identical to ppo.py) ---
    total_timesteps: int = 10_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    # --- Teacher ---
    # 'oracle' (BFS, DoorKey tasks), 'bot' (BabyAI-registered
    # tasks), 'vlm_general' (vision API, any task), or
    # 'llm_general' (text-only API, any task). The API teachers
    # cost money, so submit them deliberately; they run fine on
    # compute nodes, which do have outbound internet. See
    # teachers/factory.py and the module docstring.
    teacher: str = 'bot'
    # Model for the vlm_general/llm_general teachers. Ignored by
    # oracle/bot.
    teacher_model: str = 'gpt-4.1-mini'
    # Reasoning budget ('minimal', 'low', 'medium', 'high') for the
    # gpt-5 / o-series models. Empty sends no reasoning field, which
    # the gpt-4.x models require, and which leaves a reasoning model
    # on its own default (medium).
    #
    # Reasoning tokens bill as OUTPUT tokens, so this is the largest
    # cost lever a paid teacher has -- but it is a competence lever
    # first. Measured on doorkey_8x8 with gpt-5-mini, 4 seeds
    # (scripts/eval_teacher.py --reasoning-effort ...):
    #
    #   unset/medium  4/4 solved, 18.5 mean steps, $0.00132/call
    #   low           4/4 solved, 18.5 mean steps, $0.00071/call
    #   minimal       0/4 solved -- the teacher oscillates between
    #                 two turn actions and never commits to forward
    #
    # 'low' is therefore the setting for DoorKey: identical
    # competence, 1.8x cheaper. The cliff is between minimal and
    # low, not between low and medium. Do NOT carry that across to a
    # harder task without re-probing -- KeyCorridor is exactly where
    # the previous model's competence collapsed, see
    # docs/teacher_competence_2026-07-24.md.
    teacher_reasoning_effort: str = ''
    # Query the teacher only every k-th rollout step (1 = every
    # step). Unqueried steps simply carry no distillation target
    # (the mask machinery already handles that), so this thins the
    # supervision rather than changing its meaning. It exists to
    # cap API cost for the paid teachers -- cost scales ~1/k --
    # and should stay 1 for the free oracle/bot.
    query_interval: int = 1
    # Fixed uniform clock slots, copied from the archived MultiRoom
    # scheduler. Opt-in only; skipped terminal slots are not replaced.
    uniform_queries: bool = False
    # Common-clock scheduling (advising/clock_queries.py): 'random' or
    # 'entropy' picks one eligible env at each of query_budget shared,
    # state-independent times. Opt-in only; 'none' leaves every legacy
    # schedule unchanged.
    query_clock: str = 'none'
    # Spacing of the clock's query times: 'random' (unchanged) or
    # 'regular' (evenly spread over the same active window).
    query_clock_spacing: str = 'random'
    # Frozen LLM rule bank for --teacher rule_bank (path + SHA-256).
    rule_bank: str = ''
    rule_bank_sha256: str = ''
    # Online LLM rules (--teacher llm_rules_online): paid calls spent during
    # training, consultations plus (when enabled) blind checks.
    online_rule_calls: int = 0
    online_rule_blind: bool = True
    # Journal rows only for paid calls, for a teacher that also advises for
    # free at every step (one row per step would be gigabytes per run).
    journal_paid_only: bool = False
    # Hide free action/set lookups from the advisor so a consultation
    # is counted before mistake filtering, as with an API teacher.
    advisor_no_teacher_peek: bool = False

    # --- Fixed query windows (the distributed schedule) ---
    # 0 keeps the advisor's own timing, which is what every run before
    # this used. A positive value replaces "whenever the advisor feels
    # like it" with a list of rollout indices resolved before training
    # starts, so the paid comparison can state where it spent money and
    # an audit can check it. See advising/schedule.py.
    query_windows: int = 0
    # Vector steps at the head of each selected rollout during which
    # consultation is permitted. Slots per window = this x num_envs.
    query_window_steps: int = 10
    # Fraction of nominal training the windows span. Must not reach
    # past distill_cutoff, or the last windows buy advice the loss
    # has already switched off.
    query_window_fraction: float = 0.75
    # Price per million embedding tokens, for spend accounting. Left
    # at 0 because a hardcoded rate goes stale silently; pass the
    # rate in force on the day and the summary reports dollars. When a
    # budget ledger is in use the rate comes from the price table
    # instead and this is ignored.
    embed_price_per_million: float = 0.0

    # --- Shared spending limit ---
    # Path to the pooled allowance every paid run draws on. Empty
    # disables budgeting entirely, which is correct for the free
    # oracle/bot teachers and is what every earlier run did. When set,
    # the run reserves its whole worst case up front and refuses to
    # start if the pool cannot cover it -- a run stopped by a cap
    # halfway through has bought an arm nothing can be compared to.
    budget_ledger: str = ''
    price_table: str = 'configs/prices.json'
    # Ceilings that make one request's cost bounded. Both are required
    # under a ledger; an unbounded request cannot be reserved for.
    # 3500 output sits ~40% above the ~2500 the smoke actually spent
    # per call, nearly all of it hidden reasoning. Too tight a ceiling
    # is not free caution: a truncated response fails to parse, which
    # bills for the call and returns an abstention. Watch
    # teacher_total_abstains and raise this if truncation appears.
    max_output_tokens: int = 3500
    max_input_tokens: int = 4000
    # Hard cap on attempts per consultation, covering the client's
    # internal retries and the outer cold-start retry.
    max_attempts: int = 3
    # Operational stop, separate from the scientific schema-validity gate.
    # Zero disables it for older experiments and free teacher runs.
    consultation_failure_limit: float = 0.0
    consultation_failure_min_samples: int = 40
    # Attempts are reserved IN FULL, not in expectation, and the SDK's
    # own retries are disabled under a budget so this cap is the whole
    # of it. Reserving an expected number was tried and was wrong in
    # kind: an expectation is not a guarantee, and with the client
    # retrying underneath the outer cap one consultation was
    # demonstrated reaching a transport nine times. The honest
    # consequence is a smaller affordable grid.
    # Upper bound on tokens in one explanation, for the embedding
    # reservation.
    max_embed_tokens: int = 512

    # --- Explanation auxiliary head (R2/R3/R4) ---
    # 'none'     R0/R1: no head, no auxiliary loss.
    # 'correct'  R2: predict the teacher's own explanation embedding.
    # 'shuffled' R3: predict a within-phase donor's embedding.
    # 'detached' R4: correct target, but the head reads stop_gradient(h),
    #            so the auxiliary gradient never reaches shared features.
    explanation: str = 'none'
    # Separate simulator-only diagnostic; never substitutes LLM targets.
    consequence: str = 'none'
    consequence_coef: float = 0.1
    # Cap the simulator copies a dense-advice run pays for. Zero keeps
    # the unlimited behaviour every earlier run had.
    consequence_target_budget: int = 0
    # Train the head only where the permutation actually changes a
    # target, so treatment and placebo differ on every trained sample.
    consequence_changed_only: bool = False
    # Truncate an engineering run without rescaling training schedules.
    diagnostic_stop_rollouts: int = 0
    # New protocols opt in; historical experiments keep their old path.
    teacher_stream: bool = False
    action_reference: str = ''
    rationale_queries: bool = True
    structured_explanations: bool = False
    # llm_general only: '' keeps the prompt unchanged; 'cite' asks for
    # the one cell the advice rests on; 'cite_view' also shows the
    # teacher the student's view. Read by advice_evidence_gate.
    teacher_evidence: str = ''
    explanation_target: str = 'embedding'
    lambda_aux: float = 0.1
    # Save aligned rollout features/state and auxiliary update evidence.
    # Observation only: no additional teacher query or optimizer step.
    audit_explanations: bool = False
    # Initialization evidence without large aligned per-frame records.
    record_initial_policy: bool = False
    # Optional startup guard for a reused same-seed control. Checked before
    # the first teacher request so a mismatched pair cannot spend API money.
    expected_initial_policy_sha256: str = ''
    # Free action-only sweeps may retain counters without per-call text.
    offline_summary_only: bool = False
    embed_model: str = 'text-embedding-3-small'
    # Dimension of the frozen encoder. Recorded in the manifest
    # because vectors from another model are another dataset.
    embed_dim: int = 1536
    embed_cache: str = ''
    # Seed for the R3 donor draw, kept separate from the training seed
    # so the corrupted mapping is reproducible independently of the run.
    shuffle_seed: int = 0
    # HARD ceiling on this run's teacher spend, in dollars. 0
    # disables the cap (the default, for free teachers where there
    # is nothing to cap). Once total_cost.dollars reaches this, the
    # teacher stops being queried for the REST of the run --
    # exactly as if the distillation schedule had cut off early --
    # and training continues as pure PPO for free. This is a real,
    # code-enforced ceiling: query_interval and total_timesteps only
    # estimate cost in advance, and an estimate can be wrong (a
    # longer-than-expected reasoning response, a retry, a state more
    # complex than the ones used to calibrate the estimate). This
    # flag is what actually prevents a run from overspending
    # regardless of whether the estimate was right.
    max_cost_dollars: float = 0.0

    # --- Distillation schedule (LLM4Teach's shape, verified from
    # their code: decay to a floor, then a hard cutoff) ---
    # Initial coefficient. LLM4Teach used 10.0, which dominates the
    # early loss (their policy loss is O(0.01-0.1)): early training
    # is deliberately teacher-led, without ever touching the
    # executed actions.
    distill_coef_start: float = 10.0
    # Keep the historical labeled mean unless a protocol opts in.
    distill_normalization: str = 'labeled'
    # Floor the coefficient decays to (theirs: 0.1). Keeps a light
    # pull toward the teacher until the cutoff.
    distill_coef_min: float = 0.1
    # Fraction of training over which the coefficient decays
    # linearly from start to the floor.
    distill_fraction: float = 0.5
    # Fraction of training after which the term (and all teacher
    # queries) stop entirely. The remaining run is pure PPO, so
    # every run measures its own teacher-off retention.
    distill_cutoff: float = 0.75
    # RLingua-style execution (fix-wave addendum 8; Chen et al., RA-L
    # 2024): where the teacher gives a label, EXECUTE its action instead
    # of the student's with this probability, annealed exponentially to
    # 1% of it by `execute_teacher_fraction` of training and 0 from the
    # distillation cutoff on. 0 (the default) is the update-only channel
    # above: nothing the teacher says reaches the environment. The
    # imitation term is unchanged either way, so the two differ only in
    # whether the teacher acts. An executed action is stored with the
    # student's own log-probability of it, as an override PPO must; the
    # update's log-ratio clamp bounds the resulting importance ratios.
    execute_teacher_start: float = 0.0
    execute_teacher_fraction: float = 0.5
    # RLingua baseline (Chen et al., RA-L 2024, Algorithm A-1 and Table
    # A-II), default off. A COMPLETE controller written once by an LLM
    # (a frozen file, teachers/minigrid/rlingua_controller.py) acts
    # instead of the student with probability
    # p = rlingua_p0 * rlingua_decay ** transitions; its transitions fill
    # a persistent buffer used only for behavior cloning (weight
    # rlingua_bc_coef, constant, sampled every update as RLingua samples
    # its R_LLM buffer), while the PPO surrogate, value and entropy terms
    # use only the student's own transitions (RLingua's R_RL). Exclusive
    # of rule-bank guidance and of the execution heuristic above.
    rlingua_controller: str = ''
    rlingua_controller_sha256: str = ''
    rlingua_variant: str = 'full'
    rlingua_p0: float = 0.25
    rlingua_decay: float = 0.999999
    rlingua_bc_coef: float = 1.0
    rlingua_buffer: int = 1_000_000
    # Fraction of training to run WITHOUT the teacher before it
    # switches on (the student explores alone first, pure PPO). 0
    # keeps the teacher on from the start (the original schedule);
    # e.g. 0.3 means "let the student flail for 30% of training,
    # then bring the teacher in." The experiment knob for "what if
    # the teacher doesn't start from the beginning?"
    distill_delay_frac: float = 0.0

    # --- Advice budgeting (advising/) ---
    # `--query-interval` above spends the teacher on a fixed clock:
    # every Nth step, regardless of whether that state is worth
    # labelling. These flags replace the clock with a policy.
    #
    # 'unlimited' (the default) leaves the clock in charge and
    # reproduces every run recorded before budgeting existed.
    advisor: str = 'unlimited'
    # Opt in prospectively: query ordering must not advance PPO's
    # NumPy minibatch stream. False retains the historical path.
    advisor_rng_isolation: bool = False
    # Exercise teacher-free scheduling with virtual budget slots,
    # zero teacher instances, zero consultations and zero labels.
    advisor_sham: bool = False
    # Cap on labels bought over the whole run. Note what a budget
    # BUYS is different here than in ppo_teacher.py: there, advice is
    # an intervention consumed on the step it is given; here it is a
    # labelled example that keeps contributing to the cross-entropy
    # term. See advising/replay.py.
    advice_budget: int = 0
    query_budget: int = 0
    importance_threshold: float = 0.0
    # 'entropy' (normalized, ports across tasks), 'max_prob',
    # 'top2_gap', 'entropy_nats', 'logit_gap', 'none'. See
    # advising/policy.py.
    importance_source: str = 'none'
    # Target fraction of environment steps to label. Leave at 0 and
    # supply an advice_budget instead: the advisor then paces itself
    # to spend exactly the budget across the run, which keeps the
    # budget the controlled variable when comparing strategies.
    advice_rate: float = 0.0
    # 0 keeps Torrey & Taylor's exact-action mistake test; above 0
    # switches to "the student's policy puts less than this much
    # probability on the teacher's endorsed actions", which is the
    # test suited to a student that samples.
    mistake_threshold: float = 0.0
    predictor: str = 'count'
    # Free teacher-action lookup: 'none', 'blind', 'exact',
    # 'surrogate'. See advising/peekable.py.
    peek: str = 'none'
    peek_trust: float = 0.9

    # --- Advice replay (advising/replay.py) ---
    # Keep bought labels and replay them into the distillation term
    # for the rest of training, instead of discarding them with the
    # rollout that produced them.
    #
    # This is the whole reason budgeting is worth studying on THIS
    # channel rather than on action override. An intervention buys
    # one step; a label can be reused indefinitely, so the same
    # budget is worth far more -- and the question turns into an
    # active-learning one: which states are worth paying to have
    # labelled? Off by default, since it changes what the existing
    # 0.94 result measures.
    advice_replay: bool = False
    advice_replay_capacity: int = 10000
    # Labels drawn from the buffer per minibatch. 0 uses the same
    # size as the PPO minibatch.
    advice_replay_batch: int = 0
    # Weight on the replayed cross-entropy, relative to the
    # in-rollout one. Both are scaled by the decaying `coef`, so
    # this only sets their ratio.
    #
    # Turning replay on adds a second cross-entropy term, so at 1.0
    # the total pull toward the teacher roughly DOUBLES compared
    # with a run without replay -- which confounds "does reusing
    # labels help" with "does more distillation pressure help".
    # Halve it (0.5) to hold total pressure roughly fixed and
    # isolate the reuse, or keep 1.0 to ask what the extra signal is
    # worth outright. Which of those is the right control depends on
    # the claim being made, so it is left explicit.
    advice_replay_coef: float = 1.0

    # --- Teacher-off greedy evaluation (monitoring/eval.py) ---
    # Every this many iterations, run the fixed greedy eval set.
    # 0 disables. 50 iterations = 51,200 frames at the defaults.
    eval_interval: int = 50
    # Modest count because a bad early policy times out every
    # episode, making eval cost ~ max_steps * episodes.
    eval_episodes: int = 10
    # Evaluate the SAMPLED policy alongside the greedy one, so a
    # student scoring 0.00 can be distinguished from one whose argmax
    # merely cycles in the gridworld.
    eval_sampled: bool = True
    # Extra nominal frame endpoints, rounded down to full rollouts.
    # These supplement regular evaluation without shortening training.
    eval_frame_milestones: str = ''
    # Optional isolated measurements, including 0 before learning.
    # Round UP to completed rollouts; save in a separate diagnostic
    # file so existing AUC readers retain their original endpoints.
    diagnostic_eval_frames: str = ''

    # --- Derived at runtime in train(); not set on the CLI ---
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill
    its bias, the standard PPO initialization.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Shared-CNN actor-critic, identical to the ppo.py baseline.

    The only signature difference: get_action_and_value also
    returns the raw policy logits, because the distillation term
    needs the full action distribution (not just the log-prob of
    one action) without a second forward pass.
    """

    def __init__(self, envs, symbolic=False, recurrent=False,
                 dual_value=False):
        """
        Build the encoder, optional recurrent core, and heads.

        `symbolic` selects the integer-grid input path in
        algos/nets.ObsEncoder; False keeps the pixel behaviour
        every existing run used.

        `recurrent` inserts a GRU between the trunk and the heads.
        It exists so a teacher arm can be run at the SAME
        configuration as the exploration-bonus arms: on
        keycorridor_s3r3 those reach 1.00 at symbolic+GRU and 0.00
        on RGB, so a teacher measured only on RGB is being compared
        across three variables at once.
        """

        super().__init__()

        # One shared trunk that handles both the RGB modes and the
        # symbolic grid encoding; see algos/nets.py for why the two
        # need different scaling and different kernel sizes.
        self.encoder = ObsEncoder(
            envs.single_observation_space.shape, symbolic
        )
        self.recurrent = recurrent
        self.core = RecurrentCore(512) if recurrent else None
        self.actor = layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)
        self.critic_int = (
            layer_init(nn.Linear(512, 1), std=1.0) if dual_value else None
        )

    def values_from_hidden(self, hidden):
        """
        Return task value and, when enabled, intrinsic value.
        """

        value = self.critic(hidden)
        if self.critic_int is not None:
            value = torch.cat((value, self.critic_int(hidden)), dim=-1)
        return value

    def initial_core_state(self, num_envs, device):
        """
        Return a zeroed recurrent state, or None when feed-forward.
        """

        if self.core is None:
            return None
        return self.core.initial_state(num_envs, device)

    def _encode(self, x):
        """
        Encode a batch of (N, H, W, C) uint8 images into features.
        """

        return self.encoder(x)

    def get_states(self, x, core_state=None, episode_start=None):
        """
        Encode observations and, when recurrent, carry them through
        the GRU, zeroing the hidden state at genuine episode starts.
        """

        hidden = self._encode(x)
        if self.core is None:
            return hidden, None
        return self.core(hidden, core_state, episode_start)

    def get_value(self, x, core_state=None, episode_start=None):
        """
        Return the critic's value estimate for a batch of states.

        The updated core state is discarded: a read-only bootstrap
        query must not perturb the state the next rollout step uses.
        """

        hidden, _ = self.get_states(x, core_state, episode_start)
        return self.values_from_hidden(hidden)

    def get_action_and_value(
        self, x, action=None, core_state=None, episode_start=None
    ):
        """
        Sample (or score) an action; return it with log-prob,
        entropy, value, the raw logits (for distillation), the updated
        core state, and the shared hidden features.

        `hidden` is returned so the explanation head can read exactly
        the features the actor read, from the same forward pass. A
        second forward would recompute the same numbers, but it would
        also double the trunk's cost in the auxiliary arms and make
        "the head shapes the features the actor uses" a claim about two
        separate graphs rather than one.
        """

        hidden, core_state = self.get_states(
            x, core_state, episode_start
        )
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.values_from_hidden(hidden),
            logits,
            core_state,
            hidden,
        )


def soften_advice(advice, n_actions):
    """
    Turn one Advice into a target distribution over actions.

    LLM4Teach's uncertainty-aware form: confidence mass on the
    recommended action, the remainder spread uniformly. With the
    deterministic oracle/bot (confidence 1.0) this is one-hot, and
    the CE term reduces to -log pi(a_teacher|s); with a future LLM
    teacher (confidence < 1) weak advice exerts a weak pull.

    Returns None if the teacher abstained (no recommendation), so
    the caller can mask this state out of the loss.
    """

    if advice is None or advice.action is None:
        return None
    residual = (1.0 - advice.confidence) / n_actions
    target = np.full(n_actions, residual, dtype=np.float32)
    target[advice.action] += advice.confidence
    return target


def _append_explanation_records(directory, iteration, rows, args,
                                filename='explanation_records.jsonl'):
    """
    Append one rollout's explanation records as JSON lines.

    Everything needed to audit or re-derive the auxiliary targets:
    unique transition id, the source text and its cache key, the phase,
    the donor actually used, whether the sample survived the shared
    eligibility rule, and the provenance of the target itself.

    This is a record of collected supervision, NOT a resumable
    checkpoint. Restarting training needs optimizer, RNG, environment
    and teacher-history state, none of which is here.
    """

    if not directory or not rows:
        return
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, 'a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps({
                **row.get('audit', {}),
                'iteration': iteration,
                'rollout': row.get('rollout'),
                'sample_id': row['sample_id'],
                'text_key': row.get('text_key'),
                'text': row['text'],
                'phase': row['phase'],
                'subgoal': row.get('subgoal'),
                'target_kind': args.explanation_target,
                'action_reference': args.action_reference,
                'step': row['step'],
                'env': row['env'],
                'episode': row.get('episode'),
                'global_transition': row.get('global_transition'),
                'queried_at_utc': row.get('queried_at_utc'),
                # The donor the shared eligibility rule selected, and
                # the target actually trained on. They coincide only in
                # R3. In R2 and R4 the same rule runs -- so the masks
                # match and a sample-count artefact cannot be mistaken
                # for an effect -- but the row keeps its own target, and
                # recording the donor alone would read as though the
                # target had been swapped when it was not.
                'donor_sample_id': row.get('donor_sample_id'),
                'target_sample_id': row.get('target_sample_id'),
                'donor_is_hypothetical': (
                    None if row.get('donor_sample_id') is None
                    else args.explanation != 'shuffled'
                ),
                # None, not False: R1 has no auxiliary target, so the
                # eligibility rule was never applied to this row.
                # Claiming False would assert a test that never ran.
                'usable': row.get('usable'),
                'arm': args.explanation,
                'embed_model': (
                    args.embed_model if args.explanation != 'none'
                    and args.explanation_target == 'embedding' else None
                ),
                'teacher': args.teacher,
                'teacher_model': args.teacher_model,
            }) + chr(10))


def teacher_was_consulted(asked_before, advisor):
    """
    Did this dispatch actually reach the teacher?

    The advisor may decline entirely (importance gating, pacing, a free
    surrogate), in which case the teacher learned nothing and the
    pending history must survive to the next attempt. Consultation is
    the test rather than delivered advice: a withheld answer still
    reached the teacher, and a cache hit still calls recommend().

    Extracted so the training loop and its tests run the same code.
    """

    return advisor.num_asked > asked_before


def note_executed_action(pending, action, reset_only):
    """
    Record an action the environment actually ran.

    Under Gymnasium's NEXT_STEP autoreset the action submitted on the
    tick after a termination is discarded, so reporting it to a stateful
    teacher would describe behaviour that never happened.

    Mutates and returns `pending` so a caller can chain it.
    """

    if not reset_only:
        pending.append(int(action))
    return pending


def masked_distillation_loss(ce, mask, normalization='labeled'):
    """Distinguish strength per label from strength per sampled transition."""

    if normalization == 'labeled':
        denominator = mask.sum().clamp_min(1)
    elif normalization == 'batch':
        denominator = max(1, mask.numel())
    else:
        raise ValueError('Unknown distillation normalization')
    return (ce * mask).sum() / denominator


def distill_coef(iteration, num_iterations, args):
    """
    The kickstarting coefficient for this iteration.

    Linear decay from distill_coef_start to distill_coef_min over
    distill_fraction of training, held at the floor, then dropped
    to exactly 0 at distill_cutoff -- the schedule shape verified
    from LLM4Teach's released code (decay + floor + hard cutoff).

    An optional distill_delay_frac delays the teacher's onset: for
    the first delay fraction of training the coefficient is 0 (the
    student explores on its own, pure PPO), and the whole
    decay+cutoff schedule then plays out over the remaining
    training. delay=0 (the default) reproduces the original schedule
    exactly. This is the knob for the "what if the teacher doesn't
    start from the beginning?" experiment.
    """

    if not getattr(args, 'guidance', True):
        return 0.0
    progress = (iteration - 1) / max(1, num_iterations)

    # Delay the teacher's onset: nothing happens until `delay` of
    # training has elapsed, then progress is rescaled so the full
    # start->floor->cutoff schedule runs over the post-delay
    # portion. getattr keeps this backward-compatible for any Args
    # (e.g. the BabyAI trainer's) that has not added the field.
    delay = getattr(args, 'distill_delay_frac', 0.0)
    if progress < delay:
        return 0.0
    progress = (progress - delay) / max(1e-8, 1.0 - delay)

    if progress >= args.distill_cutoff:
        return 0.0
    frac = min(1.0, progress / max(1e-8, args.distill_fraction))
    span = args.distill_coef_start - args.distill_coef_min
    return args.distill_coef_start - span * frac


def student_mean(x, mask):
    """Mean over the student's own steps (RLingua's R_RL); plain mean
    when no step mask is in use."""
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp(min=1.0)


def extrinsic_gae(rewards, values, dones, next_value, next_done, gamma,
                  gae_lambda, keep=None):
    """GAE exactly as in ppo.py. `keep[t]` (1 where the student chose the
    action at step t) stops each trace before a step another policy chose,
    so an advantage bootstraps from the value there instead of summing
    that policy's actions (a tree-backup cut); None changes nothing."""
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = (
            rewards[t]
            + gamma * nextvalues * nextnonterminal
            - values[t]
        )
        trace = nextnonterminal
        if keep is not None and t < num_steps - 1:
            trace = nextnonterminal * keep[t + 1]
        advantages[t] = lastgaelam = (
            delta
            + gamma
            * gae_lambda
            * trace
            * lastgaelam
        )
    return advantages


def execute_probability(iteration, num_iterations, args):
    """RLingua-style probability of executing the teacher's labelled action.

    p = start * 0.01 ** (progress / execute_teacher_fraction): exponential
    annealing (RLingua anneals its controller's sampling probability
    exponentially), at 1% of the start by execute_teacher_fraction of
    training, and exactly 0 wherever the distillation coefficient is 0,
    so execution never outlives the teacher's labels.
    """
    start = getattr(args, 'execute_teacher_start', 0.0)
    if start <= 0.0 or distill_coef(iteration, num_iterations, args) <= 0.0:
        return 0.0
    progress = (iteration - 1) / max(1, num_iterations)
    fraction = max(1e-8, getattr(args, 'execute_teacher_fraction', 0.5))
    return float(start * 0.01 ** (progress / fraction))


def advising_horizon(args):
    """Environment-step index where the final teacher-active rollout ends.

    Advisor pacing is expressed against absolute environment steps. Using
    total_timesteps here leaves a finite budget unspent when distillation
    closes early, because the advisor believes it has the teacher-off tail in
    which to place labels. The end of the last positive-coefficient rollout is
    the actual runway. Delayed starts remain absolute because note_env_steps
    also advances while the teacher is off.
    """
    active = [iteration for iteration in range(1, args.num_iterations + 1)
              if distill_coef(iteration, args.num_iterations, args) > 0.0]
    return (max(active) * args.batch_size if active else 0)


def train(args, auxiliary=None):
    """
    Train a distillation-guided PPO agent and log to TensorBoard.

    The skeleton is algos/ppo.train; the additions are (1) teacher
    queries + target storage during rollout, (2) the annealed CE
    term in the update, (3) the shared teacher-off greedy eval.
    """

    from algos.advice_control import ShamAdvisor, validate_sham

    validate_sham(args, auxiliary)
    if args.advisor_no_teacher_peek and (
            args.peek != 'none' or args.importance_source == 'teacher_q'
            or args.teacher not in ('oracle', 'bot', 'door_bfs')):
        raise ValueError('No-peek advising requires a free teacher and '
                         'no teacher-side lookups')
    from teachers.controlled_advice import advisor_teacher
    if args.uniform_queries and (
        not args.guidance or args.query_budget <= 0 or args.query_windows
        or args.query_interval != 1
        or args.action_reference or args.advisor != 'unlimited'
        or args.importance_source != 'none' or args.advice_rate
    ):
        raise ValueError('Uniform query configuration is invalid')
    if args.query_clock != 'none' and (
        args.query_clock not in ('random', 'entropy')
        or not args.guidance or args.query_budget <= 0
        or args.uniform_queries or args.query_windows
        or args.query_interval != 1 or args.action_reference
        or args.advisor not in ('unlimited', 'mistake')
        or args.importance_source != 'none' or args.advice_rate
        or args.importance_threshold != 0.0
        or not args.advisor_rng_isolation or args.advisor_sham
        or args.teacher not in ('oracle', 'bot', 'door_bfs', 'llm_scoped')
    ):
        raise ValueError('Clock query configuration is invalid')
    if (args.teacher == 'rule_bank') != bool(args.rule_bank) or (
            args.rule_bank and (not args.guidance or args.advisor_sham)):
        raise ValueError('rule_bank teacher needs --rule-bank and guidance')
    if (args.teacher == 'llm_rules_online') != (args.online_rule_calls > 0) \
            or (args.online_rule_calls and (
                not args.guidance or args.advisor != 'unlimited'
                or args.query_budget or args.uniform_queries
                or args.query_clock != 'none' or not args.budget_ledger
                or args.advisor_sham)):
        raise ValueError('llm_rules_online needs --online-rule-calls, dense '
                         'unlimited advising and a budget ledger')
    if args.rule_bank:
        from teachers.minigrid.rule_bank import file_sha256
        if file_sha256(args.rule_bank) != args.rule_bank_sha256:
            raise ValueError('Frozen rule bank hash differs')
    if args.query_clock_spacing not in ('random', 'regular') or (
            args.query_clock_spacing != 'random' and
            args.query_clock == 'none'):
        raise ValueError('Clock spacing needs a clock and a known spacing')
    if args.consequence not in ('none', 'aligned', 'shuffled', 'detached'):
        raise ValueError('Unknown consequence diagnostic mode')
    if args.diagnostic_stop_rollouts < 0 or (
            args.diagnostic_stop_rollouts and args.consequence == 'none'):
        raise ValueError('Diagnostic truncation needs consequence mode')
    # Effect scoring restores a copied state and steps it once, which
    # is task-agnostic; the restorer is not, so only pairs whose
    # round trip has been verified are allowed through.
    consequence_pairs = {'doorkey_8x8': 'oracle', 'multiroom_n6': 'door_bfs',
                         'keycorridor_s3r3': 'bot'}
    if args.consequence != 'none' and (
            consequence_pairs.get(args.task) != args.teacher
            or not args.guidance or args.explanation != 'none'
            or args.action_reference or args.budget_ledger
            or not math.isfinite(args.consequence_coef)
            or args.consequence_coef <= 0
            or args.consequence_target_budget < 0):
        raise ValueError('Consequence learning requires a free planner on '
                         'DoorKey, MultiRoom or KeyCorridor S3R3')
    if not 0.0 <= args.advice_alias_rate <= 1.0:
        raise ValueError('advice_alias_rate must lie in [0, 1]')
    if args.advice_alias_rate and not args.guidance:
        raise ValueError('advice_alias_rate needs guidance to alias')
    from teachers.evidence import (CITATION_MODES, EVIDENCE_TEACHERS,
                                   GATE_MODES)
    if args.teacher_evidence not in CITATION_MODES:
        raise ValueError(f'teacher_evidence must be one of {CITATION_MODES}')
    if args.teacher_evidence and args.teacher != 'llm_general':
        raise ValueError('teacher_evidence applies to llm_general only')
    if (args.teacher == 'llm_general'
            and args.advice_evidence_gate not in ('none', 'consistent')
            and not args.teacher_evidence):
        raise ValueError('an llm_general evidence gate needs '
                         'teacher_evidence to request citations')
    if args.advice_evidence_gate not in GATE_MODES:
        raise ValueError(f'advice_evidence_gate must be one of {GATE_MODES}')
    if args.advice_evidence_gate != 'none' and (
            not args.guidance or args.teacher not in EVIDENCE_TEACHERS
            or args.advice_alias_rate):
        raise ValueError('advice_evidence_gate needs guidance from a teacher '
                         'that states evidence, without injected aliasing')
    if not 0.0 < args.advice_keep_fraction <= 1.0:
        raise ValueError('advice_keep_fraction must lie in (0, 1]')
    if args.rlingua_controller:
        if (args.guidance or args.execute_teacher_start
                or args.explanation != 'none' or args.consequence != 'none'
                or args.advice_replay or args.action_reference):
            raise ValueError('The RLingua baseline replaces every other '
                             'teaching channel; run it with guidance off')
        if args.rlingua_variant not in ('full', 'view'):
            raise ValueError("rlingua_variant must be 'full' or 'view'")
        if not (0.0 <= args.rlingua_p0 <= 1.0
                and 0.0 < args.rlingua_decay <= 1.0
                and args.rlingua_bc_coef >= 0.0 and args.rlingua_buffer > 0):
            raise ValueError('RLingua settings out of range')
    if args.rule_timing_diagnostics and (
            not args.guidance or args.teacher != 'rule_bank'
            or args.advisor != 'unlimited' or args.advisor_sham
            or args.advice_alias_rate or args.execute_teacher_start
            or args.advice_replay or args.action_reference
            or args.explanation != 'none' or args.consequence != 'none'):
        raise ValueError('Rule timing diagnostics require unmodified '
                         'rule-bank labels and student-only actions')
    if args.advice_keep_fraction < 1.0 and not args.guidance:
        raise ValueError('advice_keep_fraction needs guidance to dose')
    if args.imitation_weighting not in ('none', 'advisor'):
        raise ValueError('imitation_weighting must be none or advisor')
    if args.imitation_weighting == 'advisor' and (
            not args.guidance or args.advice_evidence_gate != 'none'
            or args.advice_keep_fraction < 1.0
            or args.advice_replay or args.action_reference
            or args.explanation != 'none' or args.consequence != 'none'
            or not args.advisor_alpha > 0):
        raise ValueError('ADVISOR weighting needs guidance, a positive '
                         'alpha, and no evidence gate, replay, reference '
                         'labels, explanation or consequence heads')
    if (args.advice_evidence_gate != 'none'
            or args.advice_keep_fraction < 1.0) and (
            args.advice_replay or args.action_reference
            or args.explanation != 'none' or args.consequence != 'none'):
        raise ValueError('advice_evidence_gate selects labels at rollout end '
                         'and cannot combine with replay, reference labels, '
                         'explanation or consequence heads, which read '
                         'labels during the rollout')
    if args.distill_normalization not in ('labeled', 'batch'):
        raise ValueError('distill_normalization must be labeled or batch')
    if args.teacher_stream and (args.teacher not in (
            'bot', 'oracle', 'door_bfs') or args.action_reference
            or args.query_windows or args.query_interval != 1):
        raise ValueError('teacher_stream requires an ungapped free teacher')
    if args.action_reference and (args.teacher != 'llm_general'
            or args.action_reference not in ('bot', 'oracle', 'door_bfs')
            or args.advisor != 'unlimited' or args.advice_replay
            or not args.structured_explanations):
        raise ValueError('Action reference requires llm_general, unlimited '
                         'rationale selection, structured output and no replay')
    if (args.guidance and not args.rationale_queries
            and not args.action_reference):
        raise ValueError('Disabling rationale queries needs an action reference')
    if args.explanation_target not in ('embedding', 'subgoal'):
        raise ValueError('explanation_target must be embedding or subgoal')
    if args.explanation_target == 'subgoal':
        from teachers.controlled_advice import SUBGOALS
        if not args.structured_explanations or args.embed_dim != len(SUBGOALS):
            raise ValueError('Subgoal targets require the structured schema '
                             'and its actual target dimension')
    if args.offline_summary_only and (
        args.teacher not in ('bot', 'oracle', 'door_bfs', 'rule_bank')
        or args.explanation != 'none' or args.audit_explanations
    ):
        raise ValueError('Summary-only logging requires a free action-only '
                         'teacher and no explanation audit')
    if args.bonus not in ('none', 'count'):
        raise ValueError('distillation supports --bonus none or count')
    if args.count_observation not in ('policy', 'local_symbolic'):
        raise ValueError('count_observation must be policy or local_symbolic')
    if args.bonus != 'none' and not args.dual_value:
        raise ValueError('--bonus count requires --dual-value')
    if args.int_coef < 0 or not 0 <= args.int_gamma < 1:
        raise ValueError('intrinsic coefficient/discount is out of range')
    if not args.guidance and (
        args.peek != 'none' or args.importance_source == 'teacher_q'
        or args.advice_replay
    ):
        raise ValueError('teacher-dependent options require --guidance')
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    eval_milestones = {int(n) // args.batch_size
                       for n in args.eval_frame_milestones.split(',')
                       if n.strip()}
    if args.num_iterations < 1:
        raise ValueError('total_timesteps must cover at least one rollout')
    diagnostic_points = diagnostic_milestones(
        args.diagnostic_eval_frames, args.batch_size, args.num_iterations)
    if diagnostic_points and args.eval_episodes < 1:
        raise ValueError('Diagnostic evaluation requires positive episodes')

    algo_tag = f'ppo_distill_{args.teacher}'
    if args.dual_value or not args.guidance or args.bonus != 'none':
        algo_tag += (f'_g{int(args.guidance)}_{args.bonus}'
                     f'_dv{int(args.dual_value)}')
    # The explanation arm is part of a run's identity, so it belongs in
    # the name. R1-R4 share task, teacher, guidance, bonus, dual value
    # and advisor; without this they differ only by a start-time
    # timestamp. That is the same collision the budgeting block below
    # was written to prevent, and it is worse here: even the runs that
    # survive cannot be told apart without opening every summary, and
    # distinguishing R2 from R4 is the entire point of the grid.
    if args.explanation != 'none':
        algo_tag += f'_x-{args.explanation}'

    if args.experiment_id:
        if not all(c.isalnum() or c in '_-' for c in args.experiment_id):
            raise ValueError('experiment_id must contain letters/digits/_/-')
        algo_tag += f'_{args.experiment_id}'

    # Fold the budgeting configuration into the run name. Without it
    # every arm of a sweep is called ppo_distill_oracle and they
    # differ only by a start-time timestamp -- so arms launched in
    # the same second land in the SAME directory and overwrite each
    # other, and even the survivors cannot be grouped by arm without
    # opening every summary. A six-arm sweep lost 23 of 30 runs to
    # exactly this.
    #
    # Only added when budgeting is in use, so runs recorded before
    # it existed keep their original names.
    if args.advisor != 'unlimited':
        parts = [args.advisor]
        if args.importance_source != 'none':
            parts.append(f'i-{args.importance_source}')
        if args.advice_budget:
            parts.append(f'b{args.advice_budget}')
        if args.mistake_threshold > 0.0:
            parts.append(f'mt{args.mistake_threshold:g}')
        if args.peek != 'none':
            parts.append(f'pk-{args.peek}')
        if args.teacher_reasoning_effort:
            parts.append(f'r-{args.teacher_reasoning_effort}')
        if args.advice_replay:
            parts.append(f'rp{args.advice_replay_coef:g}')
        algo_tag = f'{algo_tag}_' + '_'.join(parts)

    run_name = f'{args.task}__{algo_tag}__{args.seed}__{int(time.time())}'

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

    repo_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    run_dir = os.path.join(repo_root, 'results', 'runs', run_name)
    # Explanation evidence lives beside the run's own artifacts.
    explanation_dir = os.path.join(run_dir, 'explanations')
    writer = SummaryWriter(run_dir)
    writer.add_text(
        'hyperparameters',
        '|param|value|\n|-|-|\n'
        + '\n'.join(f'|{k}|{v}|' for k, v in vars(args).items()),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # Cap CPU threads to the SLURM allocation so many array jobs
    # sharing one node do not oversubscribe it -- see algos/ppo.py's
    # seeding block for the full reasoning. No-op outside SLURM.
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        torch.set_num_threads(int(slurm_cpus))

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # Keep the SyncVectorEnv reference so each sub-env's unwrapped
    # MiniGrid state can be handed to the teacher (the student only
    # ever sees pixels).
    sync_envs = gym.vector.SyncVectorEnv(
        [
            make_thunk(
                args.task,
                args.seed,
                i,
                obs_mode=args.obs_mode,
                agent_view_size=args.agent_view_size,
                obs_tile_size=args.obs_tile_size,
                obs_target_size=args.obs_target_size,
            )
            for i in range(args.num_envs)
        ]
    )
    envs = RecordEpisodeStatistics(sync_envs)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), 'this algorithm only supports discrete actions'
    n_actions = int(envs.single_action_space.n)

    # --- Teachers: one instance per parallel env ---
    # The BFS oracle is stateless (one instance would do), but the
    # bot carries per-episode plan state, so per-env instances are
    # required; using them for both keeps the code uniform.
    # vlm_general has no task restriction -- it is the only teacher
    # keycorridor_s6r3 can use, since it has no offline oracle.
    env_id = TASKS[args.task]
    if (
        args.guidance and args.teacher in ('oracle', 'llm')
        and 'doorkey' not in args.task
    ):
        raise ValueError(
            f'--teacher {args.teacher} reads the DoorKey state '
            'tuple and so only supports doorkey_* tasks; use '
            '--teacher bot for BabyAI tasks or --teacher '
            'llm_general / vlm_general for anything else (see '
            'teachers/factory)'
        )
    if args.guidance and args.teacher == 'bot' and not hasattr(
        sync_envs.envs[0].unwrapped, 'instrs'
    ):
        raise ValueError(
            f'task {args.task!r} is not a BabyAI level (no '
            '.instrs); the bot teacher cannot advise it -- use '
            '--teacher vlm_general or llm_general instead'
        )
    if (args.guidance and args.teacher == 'llm_general'
            and args.query_interval != 1):
        raise SystemExit(
            'query_interval > 1 is unsafe for llm_general: it keeps a '
            'per-episode action history and marks an action "blocked" by '
            'comparing consecutive queried states. Skipping steps makes '
            'that comparison span the gap, so the history block in the '
            'prompt describes a trajectory that never happened. Use '
            'query_interval=1, or an advisor budget, which gates on '
            'whether a state is worth labelling rather than on a clock.'
        )

    # NOTE 2026-09-09: llm_general is NOT stateless -- it keeps a
    # per-episode action history that is folded into its prompt AND
    # its cache key. The loop below now hands it every action the
    # student actually executed since the previous consultation, which
    # is a prerequisite for sparse querying but NOT a validation of it.
    # Sparse fixed querying stays unavailable until skipped queries,
    # rollout boundaries and resets are covered by integration tests
    # and the R schedule is checked; the guard immediately following
    # this comment is deliberately left in force.
    if args.explanation not in ('none', 'correct', 'shuffled', 'detached'):
        raise ValueError(
            f'--explanation must be none/correct/shuffled/detached, '
            f'got {args.explanation!r}'
        )
    if args.explanation != 'none' and not args.guidance:
        raise ValueError(
            'an explanation arm needs a teacher to explain: pass '
            '--guidance, or use --explanation none for R0.'
        )

    if args.guidance and args.teacher == 'bot' and args.query_interval != 1:
        raise ValueError(
            'query_interval > 1 is only valid for teachers that either '
            'carry no state (oracle, vlm_general) or are told about '
            'every skipped step (llm_general): the bot '
            'tracks the student one executed action at a time via '
            'replan(last_action), so skipping steps silently '
            'desynchronizes its plan. The bot is free anyway -- '
            'there is no cost to cap.'
        )
    # strict=False: the API teachers otherwise raise on a single
    # bad call (a transient 403, a request timeout, malformed JSON
    # -- all confirmed to happen on Aleph, see results/slurm/
    # eval_teacher_*.out) and that exception is not caught anywhere
    # between here and train()'s own top level, so it kills the
    # whole job -- up to 2 days of progress (submit_cc.sh's --time)
    # lost over one flaky query. strict=False makes that one query
    # an abstain (soften_advice(None, ...) below already handles a
    # None advice target) instead of a crash; ignored by the
    # offline teachers (oracle, bot, door_bfs), which have no API
    # call to fail this way.
    # Online rules: every environment's teacher shares one growing rule
    # bank, one call budget and one schedule of consultation times.
    online_rules = None
    if args.guidance and args.teacher == 'llm_rules_online':
        from teachers.minigrid.llm_rules_online import OnlineRuleState
        online_rules = OnlineRuleState(
            args.online_rule_calls, args.total_timesteps, args.seed,
            blind=args.online_rule_blind)
    teachers = [
        make_teacher(
            args.teacher, env_id, args.seed + i,
            model=args.teacher_model, strict=False,
            reasoning_effort=args.teacher_reasoning_effort,
            rule_bank=args.rule_bank, online_rules=online_rules,
        )
        for i in range(args.num_envs)
    ] if (args.guidance and args.rationale_queries
          and not args.advisor_sham) else [None] * args.num_envs
    for teacher in teachers:
        if teacher is not None and args.structured_explanations:
            teacher.structured_explanations = True
        if teacher is not None and args.teacher_evidence:
            teacher.evidence_citations = args.teacher_evidence
    references = [make_teacher(args.action_reference, env_id, args.seed + i)
                  for i in range(args.num_envs)] if (
                      args.guidance and args.action_reference) else []
    reference_fresh = [True] * args.num_envs
    reference_calls = 0
    reference_labels = 0
    reference_cost = Cost()
    query_rng = np.random.default_rng(args.seed + 90_117)
    # Addendum 8: its own stream, so execution draws move no other RNG.
    execute_rng = np.random.default_rng(args.seed + 71_003)
    # An advisor skips queries by design, which desynchronizes the
    # bot for exactly the reason `query_interval > 1` is refused
    # above: its plan advances one executed action at a time.
    if (args.guidance and args.teacher == 'bot'
            and args.advisor != 'unlimited' and not args.teacher_stream):
        raise ValueError(
            'advice budgeting skips teacher queries, which '
            'desynchronizes the stateful bot the same way '
            'query_interval > 1 does. Use a stateless teacher '
            '(oracle, vlm_general, llm_general); the bot is free '
            'anyway, so there is no spend to budget.'
        )

    # Optional free lookup of the teacher's action, which is what
    # lets mistake correcting decide BEFORE paying. See
    # advising/peekable.py; the wrapper refuses teachers that carry
    # state between calls.
    if args.peek not in ('none', 'blind', 'exact', 'surrogate'):
        raise ValueError(
            f'--peek {args.peek!r} is not valid; expected none, '
            f'blind, exact, or surrogate.'
        )
    if args.peek != 'none':
        teachers = [
            PeekableTeacher(
                t,
                num_actions=n_actions,
                predictor_kind=args.predictor,
                trust=args.peek_trust,
                exact_only=args.peek in ('exact', 'blind'),
                recall=args.peek != 'blind',
                use_inner_peek=False,
            )
            for t in teachers
        ]

    policy_importance = {
        'none': None,
        'entropy': normalized_entropy_importance,
        'entropy_nats': policy_entropy_importance,
        'max_prob': max_prob_importance,
        'top2_gap': top2_gap_importance,
        'logit_gap': logit_gap_importance,
    }
    # Torrey & Taylor's own measure, and the only teacher-SIDE one
    # here: max_a Q(s,a) - min_a Q(s,a) on the teacher's converged
    # values. On a shortest-path task those values are just negated
    # remaining distance, so the BFS oracle supplies the paper's
    # quantity exactly. It reads large at bottlenecks -- the doorway,
    # the pickup moment, a junction where the wrong turn costs
    # twenty steps -- and near zero in open corridors where every
    # move costs the same.
    #
    # Every other option is student-side: they measure whether the
    # STUDENT is unsure, which is Clouse (1996), not Torrey. Only
    # this one can test her actual claim.
    #
    # It requires a teacher with a value function, which means the
    # oracle. That restriction is not incidental: using it with an
    # LLM teacher would smuggle a planner into a run whose whole
    # premise is that a language model is all you have.
    if args.importance_source == 'teacher_q':
        if args.teacher != 'oracle':
            raise ValueError(
                'teacher_q importance comes from a planner\'s value '
                'function, so it is only available with --teacher '
                'oracle. Using it with an API teacher would give '
                'that run planner-derived information it is meant '
                'not to have. Use entropy or top2_gap instead.'
            )
        importance_fn = make_teacher_q_importance(teachers[0])
    elif args.importance_source in policy_importance:
        importance_fn = policy_importance[args.importance_source]
    else:
        raise ValueError(
            f'--importance-source {args.importance_source!r} is not '
            f'available here; expected teacher_q or one of '
            f'{sorted(policy_importance)}.'
        )

    # The importance signal reports every state as maximally important
    # until it has seen enough values to rank them, so everything
    # passes during warmup. That is harmless when warmup is a small
    # slice of the budget and fatal when it is not: at budget 50 a
    # 100-step warmup would spend the whole budget before the signal
    # meant nothing at all, turning importance advising into early
    # advising while still calling itself importance advising.
    #
    # Capped at a quarter of the budget for that reason. The default
    # of 100 is unchanged whenever the budget is large enough to
    # afford it (400 or more), so previous runs reproduce.
    warmup = 100
    if args.advice_budget:
        warmup = max(5, min(100, args.advice_budget // 4))

    advisor = make_advisor(
        args.advisor,
        num_actions=n_actions,
        importance_warmup=warmup,
        advice_budget=args.advice_budget,
        query_budget=args.query_budget,
        threshold=args.importance_threshold,
        importance_fn=importance_fn,
        target_rate=(
            args.advice_rate if args.advice_rate > 0.0 else None
        ),
        # The clock already spreads queries over time; budget pacing
        # would raise the threshold above the constant importance and
        # block every clock consultation.
        horizon=(None if args.query_clock != 'none'
                 else advising_horizon(args)),
        mistake_fn=(
            probability_mistake(args.mistake_threshold)
            if args.mistake_threshold > 0.0
            else None
        ),
        predictor_kind=args.predictor,
    )
    if args.advisor_sham:
        advisor = ShamAdvisor(advisor)

    # Resolve the query schedule BEFORE training, so a configuration
    # that cannot carry it fails at startup rather than after paying
    # for part of a run.
    windows = None
    uniform_queries = None
    clock = None
    sham_uniform_slots = []
    if args.query_clock != 'none':
        from advising.clock_queries import ClockQueries

        clock = ClockQueries(
            args.query_clock,
            [r for r in range(args.num_iterations)
             if distill_coef(r + 1, args.num_iterations, args) > 0],
            args.num_steps, args.query_budget, args.seed,
            spacing=args.query_clock_spacing)
    if args.uniform_queries:
        from advising.uniform_queries import UniformQueries

        active = [r for r in range(args.num_iterations)
                  if distill_coef(r + 1, args.num_iterations, args) > 0]
        uniform_queries = UniformQueries(
            active, args.num_steps, args.num_envs,
            args.query_budget, args.seed)
    if args.query_windows:
        if not args.guidance:
            raise ValueError(
                '--query-windows schedules teacher consultations and '
                'needs --guidance; R0 queries nothing.'
            )
        windows = QueryWindows(
            total_timesteps=args.total_timesteps,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            num_envs=args.num_envs,
            num_windows=args.query_windows,
            window_steps=args.query_window_steps,
            guidance_fraction=args.query_window_fraction,
        )
        # No window may land in the teacher-off tail. Checked against
        # the coefficient schedule itself rather than against a second
        # copy of the 0.75 arithmetic: the two floor differently, and a
        # one-rollout disagreement would buy advice multiplied by zero.
        for rollout in windows.rollouts:
            if distill_coef(rollout + 1, args.num_iterations, args) <= 0.0:
                raise ValueError(
                    f'query window at rollout {rollout} falls in the '
                    f'teacher-off tail (distill_cutoff='
                    f'{args.distill_cutoff}, query_window_fraction='
                    f'{args.query_window_fraction}); those consultations '
                    f'would be paid for and then discarded'
                )
        print(f'query schedule: {windows.num_windows} windows x '
              f'{windows.window_steps} steps x {args.num_envs} envs = '
              f'{windows.scheduled_slots} slots, first {windows.rollouts[0]} '
              f'last {windows.rollouts[-1]} of {windows.num_rollouts} '
              f'rollouts')

    # Bought labels, retained across rollouts when enabled. Without
    # it a label is discarded with the rollout that produced it and
    # a budget buys `update_epochs` gradient steps; with it, the
    # same label keeps contributing for the rest of training.
    advice_replay = (
        AdviceReplay(
            capacity=args.advice_replay_capacity, seed=args.seed
        )
        if args.advice_replay
        else None
    )

    # Per-env bookkeeping the stateful bot needs: the action the
    # student last executed (fed to replan) and whether this env
    # just started a fresh episode (rebuild the bot's plan).
    last_actions = [None] * args.num_envs
    fresh_episode = [True] * args.num_envs
    # Actions each env actually executed since its last teacher query.
    # llm_general folds this into its prompt history and cache key, so
    # skipping it makes the teacher reason about a trajectory that
    # never happened. Cleared on every query and at episode end.
    executed_since_query = [[] for _ in range(args.num_envs)]
    # Per-env episode index, so a record says which episode it came
    # from rather than only which rollout.
    episode_ids = [0] * args.num_envs
    # Audit trail for the query schedule: how many records were kept,
    # and which rollouts actually produced consultations. The second is
    # what a reviewer checks the resolved window list against.
    expl_records_written = 0
    queried_rollouts = set()
    # Explanations collected during the current rollout, resolved into
    # embedding targets once the rollout is complete. Deferred rather
    # than embedded per step so repeated text costs one request, and so
    # the R3 donor pool is a whole rollout rather than a single frame.
    pending_expl = []
    embedder = None
    shuffler = None
    if args.explanation != 'none':
        embedder = EmbeddingProvider(
            model=args.embed_model,
            # A per-run cache by default. A shared path is faster but
            # this writer is not multi-process safe, so sharing one
            # across concurrent arms needs an external lock.
            cache_path=(
                args.embed_cache
                or os.path.join(explanation_dir, 'embedding_cache.json')
            ),
        ) if args.explanation_target == 'embedding' else None
        # Built for EVERY auxiliary arm, not just R3. Its donor mask
        # is the eligibility rule, and R2/R4 must drop exactly the
        # samples R3 drops -- otherwise the arms differ in how much
        # auxiliary supervision they received and an R2-over-R3 gap
        # could be a sample-count artefact. Only R3 uses the donor
        # TARGETS; all three use the mask.
        shuffler = PhaseShuffler(seed=args.shuffle_seed)

    # Reserve this run's worst case from the shared pool BEFORE the
    # first paid request. Everything needed to bound it is known now:
    # how many consultations may happen, what each can cost at its
    # token ceilings, how many attempts each may take, and how many
    # explanations may be embedded.
    budget = None
    budget_manifest = None
    if args.budget_ledger and args.guidance and args.teacher.startswith(
            ('llm', 'vlm')):
        prices = PriceTable.load(args.price_table)
        worst_case, budget_manifest = reservation_for(
            args,
            windows.scheduled_slots if windows is not None else None,
            prices,
        )
        if args.explanation != 'none':
            # One rate for both the reservation and the reported spend,
            # so the summary cannot disagree with the ledger.
            args.embed_price_per_million = prices.embedding_rate(
                args.embed_model)
        budget = CostLedger(args.budget_ledger)
        remaining = budget.reserve(
            run_name, worst_case,
            note=f'{args.explanation} arm, '
                 f'{budget_manifest["max_consultations"]} consultations'
        )
        budget_manifest['pool_available_after_usd'] = remaining
        # The mid-run cap becomes the reservation, so measured spend can
        # never run past what was set aside even if the bound was wrong.
        args.max_cost_dollars = (
            worst_case if args.max_cost_dollars <= 0
            else min(args.max_cost_dollars, worst_case)
        )
        # Bound the outer retry loop too. Without this the cold-start
        # retry is bounded by elapsed time, not attempts, so the
        # max_attempts factor in the reservation would be fiction.
        # Pin BOTH retry layers. The outer cold-start cap alone is not
        # how many times the provider is reached: the SDK client retries
        # underneath it and the two multiply. Disabling the inner one
        # makes attempts-per-consultation equal to the cap the
        # reservation was built on.
        os.environ['LLM_MAX_ATTEMPTS'] = str(max(1, args.max_attempts))
        os.environ['LLM_MAX_SDK_RETRIES'] = '0'
        os.environ['LLM_MAX_OUTPUT_TOKENS'] = str(args.max_output_tokens)
        os.environ['LLM_MAX_INPUT_TOKENS'] = str(args.max_input_tokens)
        os.environ['LLM_MAX_EMBED_TOKENS'] = str(args.max_embed_tokens)
        print(f'budget: reserved ${worst_case:.2f} for {run_name}; '
              f'${remaining:.2f} left in the pool')

    # The observation mode decides how the trunk reads the
    # observation, so derive it once here.
    symbolic = is_symbolic_mode(args.obs_mode)
    agent = Agent(
        envs, symbolic=symbolic, recurrent=args.recurrent,
        dual_value=args.dual_value,
    ).to(device)
    if (args.audit_explanations or args.record_initial_policy
            or args.expected_initial_policy_sha256):
        # Record actual initialization, not only an intended common seed.
        initial_digest = hashlib.sha256()
        for name, tensor in sorted(agent.state_dict().items()):
            initial_digest.update(name.encode() + b'\0')
            initial_digest.update(tensor.detach().cpu().numpy().tobytes())
        initial_policy_sha256 = initial_digest.hexdigest()
        if (args.expected_initial_policy_sha256
                and initial_policy_sha256
                != args.expected_initial_policy_sha256):
            raise ValueError(
                'Initial policy does not match the reused same-seed '
                f'control: expected {args.expected_initial_policy_sha256}, '
                f'got {initial_policy_sha256}. No API request was made.')
        with open(os.path.join(run_dir, 'initial_policy.sha256'), 'w',
                  encoding='utf-8') as handle:
            handle.write(initial_policy_sha256 + '\n')
    # The head is constructed AFTER the agent, deliberately. Building it
    # first would consume RNG draws and shift the policy's
    # initialization, so R4 would start from different weights than R1
    # for a reason that has nothing to do with the control.
    explanation_head = None
    if args.explanation != 'none':
        from algos.explanation_head import ExplanationHead
        # Built inside a FORKED RNG stream. Constructing the head draws
        # from the global generator, and without the fork every random
        # draw afterwards would differ -- including action sampling in
        # the rollout. R4 would then visit different states than R1 and
        # diverge for a reason that has nothing to do with the auxiliary
        # gradient, which is exactly the confound R4 exists to rule out.
        # The integrated equivalence test catches this; the standalone
        # harness did not, because it only checked initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 90_001)
            explanation_head = ExplanationHead(
                hidden_dim=512, embed_dim=args.embed_dim
            )
        explanation_head = explanation_head.to(device)

    # One optimizer, but note the clipping below: it covers
    # agent.parameters() ONLY. That is what keeps R4 equal to R1 --
    # if the head's gradients entered the global norm they would
    # rescale the policy's gradients by a factor that depends on the
    # auxiliary loss. Adam keeps per-parameter state, so adding the
    # head's parameters here does not perturb the policy's own updates.
    consequence_head = None
    consequence_data = None
    if args.consequence != 'none':
        from algos.consequence_head import ConsequenceHead
        from algos.consequence_training import SimulatorConsequences
        consequence_head = ConsequenceHead(
            hidden_dim=512, seed=args.seed+90_002).to(device)
        consequence_data = SimulatorConsequences(args, run_dir, device)
    trainable = list(agent.parameters())
    if consequence_head is not None:
        trainable += list(consequence_head.parameters())
    if explanation_head is not None:
        trainable += list(explanation_head.parameters())
    optimizer = optim.Adam(
        trainable, lr=args.learning_rate, eps=1e-5
    )
    # Optional frozen-lesson study; default training is unchanged.
    if auxiliary is not None:
        auxiliary.initialize(agent, run_dir, device, args)
    # ADVISOR's auxiliary actor. Built under a forked RNG so the policy's
    # initialization and random stream match the unweighted arm exactly;
    # its own optimizer keeps it out of the policy's gradient clipping.
    aux_actor = aux_optimizer = None
    advisor_stats = {'weight_sum': 0.0, 'labels': 0, 'aux_ce_sum': 0.0}
    if args.imitation_weighting == 'advisor':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(args.seed + 424_242)
            aux_actor = layer_init(
                nn.Linear(agent.actor.in_features, n_actions), std=0.01
            ).to(device)
        aux_optimizer = optim.Adam(
            aux_actor.parameters(), lr=args.learning_rate, eps=1e-5)

    # A recurrent update replays each env's contiguous sequence from
    # the state the rollout began in, so minibatches are groups of
    # ENVS rather than arbitrary timesteps; that only divides evenly
    # when num_envs does.
    if args.recurrent:
        assert args.num_envs % args.num_minibatches == 0, (
            f'recurrent minibatching needs num_envs '
            f'({args.num_envs}) divisible by num_minibatches '
            f'({args.num_minibatches})'
        )
        envs_per_batch = args.num_envs // args.num_minibatches

    obs_shape = envs.single_observation_space.shape
    count_shape = (obs_shape if args.count_observation == 'policy' else
                   (args.agent_view_size, args.agent_view_size, 3))
    exploration = (
        CountExploration(count_shape, args.num_envs, device, args)
        if args.bonus == 'count' or args.coverage_log else None
    )
    tracker = RunTracker(
        run_dir=run_dir,
        run_name=run_name,
        algo=algo_tag,
        args=args,
        obs_shape=obs_shape,
        action_space=envs.single_action_space,
        unwrapped_env=sync_envs.envs[0].unwrapped,
        device=device,
        extra=runtime_metadata(),
    )
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + obs_shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs)
        + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros_like(values)
    next_obs_buf = (torch.zeros(
        (args.num_steps, args.num_envs) + count_shape, device=device)
        if exploration else None)
    # Teacher targets q(.|s) per (step, env), plus a mask that is 0
    # where the teacher abstained, the step is a post-terminal
    # placeholder, or queries are off (post-cutoff).
    teacher_targets = torch.zeros(
        (args.num_steps, args.num_envs, n_actions)
    ).to(device)
    teacher_mask = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    # Explanation targets, one frozen embedding per transition, with
    # their own mask. Separate from teacher_mask because a consultation
    # can yield an action label without a usable explanation target --
    # the text may be missing, or R3 may have found no within-phase
    # donor for it.
    expl_targets = torch.zeros(
        (args.num_steps, args.num_envs, args.embed_dim)
    ).to(device)
    expl_mask = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)
    # Note: both explanation buffers are zeroed at the top of every
    # rollout below, alongside the action-target buffers.
    # episode_starts is `dones` shifted one step, which is what the
    # recurrent core consumes; see algos/nets.RecurrentCore for why
    # the two are not the same array.
    episode_starts = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)

    global_step = 0
    start_time = tracker.start_time
    next_obs, _ = envs.reset(seed=args.seed)
    if auxiliary is not None and hasattr(auxiliary, 'on_reset'):
        # Reward-shaping auxiliaries anchor their potential on the true
        # reset states, before the first policy action.
        auxiliary.on_reset(sync_envs, global_step)
    coverage = TrainingCoverage(sync_envs.envs) if args.coverage_log else None
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # The first observation of training genuinely follows a reset.
    episode_start = torch.ones(args.num_envs).to(device)
    core_state = agent.initial_core_state(args.num_envs, device)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    total_queries = 0
    total_abstains = 0
    # Teacher actions executed in the environment (addendum 8 only).
    teacher_executed = 0
    # RLingua baseline: the controller, its own random stream, the
    # per-rollout record of which steps it took, and the persistent
    # buffer of its transitions (RLingua's R_LLM) used for cloning.
    rlingua = None
    rl_exec = torch.zeros((args.num_steps, args.num_envs)).to(device)
    if args.rlingua_controller:
        from teachers.minigrid.rlingua_controller import RLinguaController
        rlingua = RLinguaController(
            args.rlingua_controller, args.rlingua_variant, args.num_envs,
            sha256=args.rlingua_controller_sha256 or None)
        rl_rng = np.random.default_rng(args.seed + 81_007)
        rl_bc_rng = np.random.default_rng(args.seed + 81_011)
        rl_cap = int(args.rlingua_buffer)
        rl_buf_obs = torch.zeros(
            (rl_cap,) + envs.single_observation_space.shape,
            dtype=torch.uint8)
        rl_buf_core = (torch.zeros((rl_cap, agent.core.hidden))
                       if agent.core is not None else None)
        rl_buf_start = torch.zeros(rl_cap)
        rl_buf_act = torch.zeros(rl_cap, dtype=torch.long)
        rl_buf_n = 0                       # filled entries
        rl_buf_next = 0                    # ring-buffer write position
        rl_executed = 0                    # controller actions taken
        rl_transitions = 0                 # real transitions, for the decay
        rl_bc_sum = 0.0                    # cloning loss, summed per update
        rl_bc_count = 0
    # Labels redirected by advice_alias_rate. A dense run writes no
    # journal rows, so without this the realized rate is unverifiable.
    total_aliased = 0
    # Evidence gate: a verdict per usable label during the rollout, and
    # the selection at rollout end (so the random placebo can match the
    # gate's delivered count). The teacher's answer and its journal row
    # are untouched; teachers/evidence.py documents the modes.
    from teachers.evidence import EvidenceGate
    evidence_gate = (
        EvidenceGate(args.advice_evidence_gate, args.teacher,
                     seed=args.seed + 7_654_321,
                     keep_fraction=args.advice_keep_fraction)
        if (args.advice_evidence_gate != 'none'
            or args.advice_keep_fraction < 1.0) else None)
    gate_verdicts = np.zeros((args.num_steps, args.num_envs), dtype=bool)
    # Paid teachers get one row per label (bounded by the budget); dense
    # free teachers get one row per rollout.
    gate_label_ids = {}
    gate_log_path = os.path.join(run_dir, 'evidence_gate.jsonl')
    gate_label_path = os.path.join(run_dir, 'evidence_gate_labels.jsonl')
    from monitoring.rule_timing import RuleTimingDiagnostics
    rule_timing = (RuleTimingDiagnostics(
        run_dir, (args.num_steps, args.num_envs), teachers)
        if args.rule_timing_diagnostics else None)
    # Dispatches where the advisor refused to consult at all. Not a
    # teacher failure and not a cost; kept separate for exactly that
    # reason.
    total_declined = 0
    total_cost = Cost()
    last_eval = None
    # Printed once, the first time max_cost_dollars stops queries --
    # otherwise this would print every remaining step of the run.
    budget_notice_printed = False

    consultation_journal = ConsultationJournal(
        os.path.join(run_dir, 'consultations.jsonl'),
        failure_limit=args.consultation_failure_limit,
        minimum=args.consultation_failure_min_samples,
        write_rows=not args.offline_summary_only,
        paid_only=args.journal_paid_only)

    def evaluate_student():
        """
        Measure the student with fresh environments and recurrent state.
        """

        def make_select_action(greedy):
            """
            Build an action closure, argmax or sampled.

            For a recurrent policy the closure carries a hidden
            state and exposes `.reset`, which greedy_eval calls
            after each env.reset(). Because greedy_eval runs
            episodes strictly one at a time, episode_start can
            always be zero here -- reset() has already zeroed the
            state at exactly the right moment.
            """

            state = {'core': agent.initial_core_state(1, device)}

            def reset():
                state['core'] = agent.initial_core_state(
                    1, device
                )

            def select_action(o):
                with torch.no_grad():
                    t = torch.Tensor(o).unsqueeze(0).to(device)
                    zero_start = torch.zeros(1, device=device)
                    hidden, state['core'] = agent.get_states(
                        t, state['core'], zero_start
                    )
                    logits_1 = agent.actor(hidden)
                    if greedy:
                        return int(
                            logits_1.argmax(dim=-1).item()
                        )
                    return int(
                        Categorical(logits=logits_1)
                        .sample()
                        .item()
                    )

            if agent.recurrent:
                select_action.reset = reset
            return select_action

        def make_eval_env():
            """
            Build a fresh eval env matching training's pipeline.
            """

            return build_env(
                args.task,
                seed=args.seed + 50_000,
                obs_mode=args.obs_mode,
                agent_view_size=args.agent_view_size,
                obs_tile_size=args.obs_tile_size,
                obs_target_size=args.obs_target_size,
            )

        stats = greedy_eval(
            make_select_action(greedy=True),
            make_env=make_eval_env,
            num_episodes=args.eval_episodes,
            seed_base=args.seed + 50_000,
        )
        # Greedy remains primary. Sampled evaluation distinguishes
        # failure under argmax from failure under both action rules;
        # neither metric alone establishes a particular mechanism.
        if args.eval_sampled:
            sampled_eval = greedy_eval(
                make_select_action(greedy=False),
                make_env=make_eval_env,
                num_episodes=args.eval_episodes,
                seed_base=args.seed + 50_000,
            )
            stats['sampled_success_rate'] = sampled_eval[
                'success_rate'
            ]
        return stats

    def record_diagnostic(iteration, regular_stats=None, before=None):
        """
        Measure without changing training, or reuse a regular evaluation.
        """

        if iteration not in diagnostic_points:
            return
        before = before or policy_sha256(agent)
        isolated = regular_stats is None
        rng_seed = (args.seed + 150_000 + iteration) % (2 ** 32)
        if isolated:
            with isolated_evaluation_rng(rng_seed):
                stats = evaluate_student()
        else:
            # Do not repeat a coincident legacy evaluation or change its
            # RNG behavior. Its sampled score remains the legacy score.
            stats = dict(regular_stats)
        after = policy_sha256(agent)
        if before != after:
            raise RuntimeError('Evaluation changed the student parameters')
        row = {
            'global_step': global_step,
            'iteration': iteration,
            'requested_frames': diagnostic_points[iteration],
            'teacher_on': False,
            'episodes': args.eval_episodes,
            'seed_base': args.seed + 50_000,
            'training_episodes': tracker.episode_count,
            'wall_time_sec': time.time() - start_time,
            'reused_regular_evaluation': not isolated,
            'rng_isolated': isolated,
            'rng_seed': rng_seed if isolated else None,
            'policy_sha256_before': before,
            'policy_sha256_after': after,
            **stats,
        }
        path = os.path.join(run_dir, 'diagnostic_evaluations.jsonl')
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        for key in ('success_rate', 'sampled_success_rate'):
            if key in stats:
                writer.add_scalar('diagnostics/' + key, stats[key],
                                  global_step)

    # This opt-in point is before the first rollout or optimizer step.
    record_diagnostic(0)

    query_order_draws = 0
    teacher_labels = 0
    first_label_global_step = None
    stop_iteration = (min(args.num_iterations, args.diagnostic_stop_rollouts)
                      if args.diagnostic_stop_rollouts else args.num_iterations)
    for iteration in range(1, stop_iteration + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate
            if aux_optimizer is not None:
                aux_optimizer.param_groups[0]['lr'] = (
                    frac * args.learning_rate)

        coef = distill_coef(iteration, args.num_iterations, args)
        execute_p = execute_probability(iteration, args.num_iterations, args)

        # The update replays each minibatch's stored sequence from
        # exactly the state this rollout started in, so snapshot it
        # before the rollout advances it.
        initial_core_state = (
            core_state.detach().clone()
            if core_state is not None
            else None
        )

        # Online auxiliaries track the student's observations before
        # collection. Existing snapshot auxiliaries use neither hook.
        if auxiliary is not None and hasattr(auxiliary, 'start_collection'):
            auxiliary.start_collection(iteration)

        # --- Rollout collection ---
        # Identical to vanilla PPO except that the teacher is also
        # queried on each visited state. The executed action is
        # ALWAYS the student's own sample.
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            episode_starts[step] = episode_start

            if auxiliary is not None and hasattr(auxiliary, 'observe'):
                auxiliary.observe(
                    step, global_step, next_obs, episode_start, next_done
                )

            # The RLingua buffer stores, with each controller transition,
            # the recurrent state the student entered this step with.
            core_in = core_state
            with torch.no_grad():
                # Entropy and logits used to be discarded here. The
                # advisor reads both to score how uncertain the
                # student is at this state, so they are kept -- the
                # forward pass already computed them, so this costs
                # nothing.
                (
                    action,
                    logprob,
                    entropy,
                    value,
                    logits,
                    core_state,
                    _rollout_hidden,
                ) = agent.get_action_and_value(
                    next_obs,
                    core_state=core_state,
                    episode_start=episode_start,
                )
                values[step] = value[:, 0]
                if args.dual_value:
                    int_values[step] = value[:, 1]
            actions[step] = action
            logprobs[step] = logprob

            # dones[step] (just stored) says whether the NEXT
            # observation begins a fresh episode.
            episode_start = dones[step].clone()

            # Query the teacher for every env, unless the schedule
            # has cut distillation off (then skip entirely -- with
            # a paid teacher this is also where cost stops), the
            # step falls between query-interval ticks (paid-teacher
            # cost control; unqueried steps just carry no target),
            # or this step is the post-terminal placeholder
            # gymnasium inserts before auto-reset (the env's
            # internals still hold the finished episode, so advice
            # there would be meaningless; the env ignores the
            # action anyway).
            under_budget = (
                args.max_cost_dollars <= 0
                or total_cost.dollars < args.max_cost_dollars
            )
            # Every env advances a step regardless of whether it was
            # labelled, which is the denominator any cost claim has
            # to be measured against.
            advisor.note_env_steps(args.num_envs)

            # Outside a scheduled window nothing is asked, but every
            # other bookkeeping path still runs: executed actions keep
            # accumulating (below, after the env step), episodes still
            # reset the history, and note_env_steps above still counts
            # the denominator. The teacher's view of what the student
            # did therefore stays complete across a gap of hundreds of
            # rollouts -- which is why the gate lives here and not
            # inside the teacher.
            in_window = (
                windows is None or windows.allows(iteration - 1, step)
            )
            stream = [None] * args.num_envs
            if coef > 0 and (args.teacher_stream or references):
                from teachers.controlled_advice import (
                    consult_reference, reference_target)
                for i in range(args.num_envs):
                    if next_done[i] > 0:
                        continue
                    kind = args.action_reference or args.teacher
                    source = references[i] if references else teachers[i]
                    stream[i] = consult_reference(
                        source, kind, sync_envs.envs[i].unwrapped,
                        reference_fresh[i], last_actions[i])
                    reference_fresh[i] = False
                    reference_calls += 1
                    reference_cost = reference_cost + stream[i].cost
                    dense = reference_target(stream[i], n_actions)
                    if references and dense is not None:
                        teacher_targets[step, i] = torch.from_numpy(dense)
                        teacher_mask[step, i] = 1.0
                        reference_labels += 1
            if (
                coef > 0.0
                and args.rationale_queries
                and in_window
                and step % args.query_interval == 0
                and under_budget
            ):
                # Shuffled so a global budget is not always won by
                # the lowest env index, which would teach env 0's
                # state distribution more than env 7's.
                order = (query_rng.permutation(args.num_envs)
                         if references or args.advisor_rng_isolation
                         else np.random.permutation(args.num_envs))
                query_order_draws += 1
                clock_env = None
                if clock is not None and clock.scheduled(iteration - 1, step):
                    clock_env = clock.choose(
                        iteration - 1, step,
                        [j for j in range(args.num_envs) if next_done[j] == 0],
                        entropy.detach().cpu().numpy(), n_actions, query_rng)
                for i in order:
                    i = int(i)
                    if (uniform_queries is not None and not
                            uniform_queries.allows(iteration - 1, step, i)):
                        continue
                    if clock is not None and i != clock_env:
                        continue
                    if next_done[i] > 0:
                        continue

                    # The advisor is the gate the fixed
                    # `query_interval` clock cannot be: it decides
                    # whether THIS state is worth a label, rather
                    # than labelling every Nth step whatever it
                    # happens to contain. `unlimited` approves
                    # everything, which is the original behaviour.
                    advisor.note_step(step)
                    ctx = {
                        'entropy': float(entropy[i].item()),
                        # Every action this env actually executed since
                        # the previous query. llm_general needs it to
                        # keep its prompt history truthful: the student
                        # picks its own action, so the teacher's own
                        # advice is not what happened.
                        'executed_actions': list(executed_since_query[i]),
                        'action_probs': (
                            F.softmax(logits[i], dim=-1)
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    }
                    student_action_i = int(action[i].item())
                    # The advisor may decline to consult at all
                    # (importance gating, pacing, a free surrogate).
                    # Only a real consultation may clear the pending
                    # history, so record the counter first and compare
                    # afterwards. Consultation is the test, not whether
                    # the answer was delivered: a withheld answer still
                    # reached the teacher, and a cache hit still calls
                    # recommend().
                    asked_before = advisor.num_asked

                    u_i = sync_envs.envs[i].unwrapped
                    if args.advisor_sham:
                        selected = advisor.schedule(
                            u_i, student_action_i, ctx, global_step)
                        if selected and uniform_queries is not None:
                            sham_uniform_slots.append(
                                ((iteration - 1) * args.num_steps + step)
                                * args.num_envs + i)
                        continue
                    if args.teacher_stream:
                        from teachers.controlled_advice import CurrentAdvice
                        advice, advice_cost = advisor.advise(
                            advisor_teacher(CurrentAdvice(stream[i]),
                                            args.advisor_no_teacher_peek), u_i,
                            student_action_i, ctx)
                    elif args.teacher == 'oracle':
                        advice, advice_cost = advisor.advise(
                            advisor_teacher(teachers[i],
                                            args.advisor_no_teacher_peek),
                            extract_doorkey_state(u_i),
                            student_action_i,
                            ctx,
                        )
                    elif args.teacher == 'bot':
                        # The bot is refused above when budgeting is
                        # on, so the advisor here is always
                        # `unlimited` and cannot break its plan
                        # chain by skipping a step.
                        ctx.update(
                            {
                                'new_episode': fresh_episode[i],
                                'last_action': last_actions[i],
                            }
                        )
                        advice, advice_cost = advisor.advise(
                            advisor_teacher(teachers[i],
                                            args.advisor_no_teacher_peek),
                            u_i, student_action_i, ctx
                        )
                    elif args.teacher == 'vlm_general':
                        # The teacher reasons over the FULL rendered
                        # map (not the agent's partial pixels) --
                        # same "a teacher should see at least as
                        # much as the planner it stands in for"
                        # argument teachers/minigrid/vlm.py makes.
                        ctx.update(
                            {
                                'image': u_i.get_frame(
                                    highlight=False, tile_size=32
                                ),
                                'mission': getattr(u_i, 'mission', ''),
                            }
                        )
                        advice, advice_cost = advisor.advise(
                            teachers[i], u_i, student_action_i, ctx
                        )
                    elif args.teacher == 'llm_general':
                        # Text-only: the teacher renders its own
                        # ASCII map from the unwrapped env, so the
                        # context needs the mission string -- and
                        # `new_episode`, because this teacher keeps a
                        # per-episode action history. Its own docstring
                        # warns that a caller which never passes it
                        # gets a history that leaks across episode
                        # boundaries, which is exactly the bug this
                        # branch had: every rationale after episode 1
                        # was conditioned on actions from a previous
                        # episode.
                        ctx['mission'] = getattr(u_i, 'mission', '')
                        ctx['new_episode'] = fresh_episode[i]
                        if references and stream[i].action is not None:
                            ctx['reference_action'] = stream[i].action
                        advice, advice_cost = advisor.advise(
                            teachers[i], u_i, student_action_i, ctx
                        )
                    elif args.teacher == 'llm_subgoal':
                        # Hybrid teacher: the LLM selects a symbolic
                        # sub-goal and its internal BFS planner converts
                        # that active sub-goal into primitive actions. It
                        # is stateful across steps and must be told about
                        # episode boundaries, but unlike llm_general it
                        # does not build a prompt history from every
                        # executed primitive action.
                        ctx['mission'] = getattr(u_i, 'mission', '')
                        ctx['new_episode'] = fresh_episode[i]
                        advice, advice_cost = advisor.advise(
                            teachers[i], u_i, student_action_i, ctx
                        )
                    elif args.teacher == 'llm':
                        # The per-step DoorKey text teacher. Unlike
                        # llm_general it carries no history between
                        # calls, which is what lets --peek reuse its
                        # answers: a recorded answer stays valid
                        # only if the same state always earns the
                        # same one. That makes this the teacher to
                        # use when the point is to measure what
                        # budgeting saves against a paid API.
                        ctx['mission'] = getattr(u_i, 'mission', '')
                        advice, advice_cost = advisor.advise(
                            teachers[i],
                            extract_doorkey_state(u_i),
                            student_action_i,
                            ctx,
                        )
                    elif args.teacher == 'rule_bank':
                        # Frozen LLM bank read from the student's own
                        # view; no planner, no API call.
                        advice, advice_cost = advisor.advise(
                            teachers[i], u_i, student_action_i, ctx)
                    elif args.teacher in ('llm_scoped', 'llm_rules_online'):
                        # Paid. llm_scoped: the offline consultation prompt
                        # on this visited state, action_now only.
                        # llm_rules_online: free advice from its growing
                        # bank, plus the scheduled paid consultations.
                        advice, advice_cost = advisor.advise(
                            teachers[i], u_i, student_action_i, ctx)
                    elif args.teacher == 'door_bfs':
                        # Stateless (a fresh BFS over the live grid
                        # every call -- teachers/minigrid/door_bfs.py
                        # keeps no memory between steps), so unlike
                        # 'bot' this needs no episode-tracking
                        # context, and unlike 'llm_general' it reads
                        # no mission text: DoorOnlyBFSTeacher.
                        # recommend() looks only at grid/agent_pos/
                        # agent_dir on the unwrapped env passed as
                        # `state`.
                        advice, advice_cost = advisor.advise(
                            advisor_teacher(teachers[i],
                                            args.advisor_no_teacher_peek),
                            u_i, student_action_i, ctx
                        )
                    else:
                        raise ValueError(
                            f'ppo_distill.py does not know how to '
                            f'query teacher {args.teacher!r} (see '
                            'teachers/factory.py)'
                        )
                    # Acknowledge the context ONLY if the teacher
                    # was actually consulted. Clearing after a skipped
                    # consultation would drop the student's actions and
                    # the new-episode notification on the floor, and the
                    # teacher would never learn either happened.
                    consulted = teacher_was_consulted(
                        asked_before, advisor
                    )
                    if consulted:
                        if uniform_queries is not None:
                            uniform_queries.record(iteration - 1, step, i)
                        fresh_episode[i] = False
                        executed_since_query[i].clear()
                    total_queries = advisor.num_asked
                    # The advisor returns the cost even when it
                    # withholds the answer: a discarded query is
                    # still billed.
                    total_cost = total_cost + advice_cost
                    if args.advice_alias_rate:
                        from teachers.controlled_advice import (
                            aliased_advice,
                        )
                        # Before the journal row, so the record shows
                        # the label the student actually received, and
                        # the cost the journal reads is the aliased
                        # one, or the flag never reaches the file.
                        # Every billed field is copied unchanged, and
                        # total_cost was summed above, so the accounts
                        # are untouched.
                        advice = aliased_advice(
                            advice, u_i, n_actions,
                            args.advice_alias_rate)
                        advice_cost = advice.cost
                        total_aliased += bool(
                            advice.cost.metadata.get('aliased'))
                    target = soften_advice(advice, n_actions)
                    if references and target is not None:
                        from teachers.controlled_advice import reference_target
                        target = reference_target(stream[i], n_actions)
                    if clock is not None:
                        clock.outcome(consulted, target is not None,
                                      global_step)
                    if consulted:
                        stop_reason = consultation_journal.record(
                            advice_cost, advice, target is not None,
                            sample_id=f'{run_name}:{iteration}:{step}:{i}',
                            rollout=iteration - 1, step=step, env=i,
                            episode=episode_ids[i],
                            student_action=student_action_i,
                            teacher=args.teacher,
                            teacher_model=args.teacher_model,
                            queried_at_utc=time.strftime(
                                '%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                        if stop_reason:
                            raise BudgetError(stop_reason)
                    # After the journal, so the record keeps the answer
                    # the teacher gave. The verdict is recorded now and
                    # applied at rollout end; the label still enters
                    # teacher_mask here as usable.
                    if rule_timing is not None and consulted:
                        rule_timing.record(
                            step, i, teachers[i],
                            student_action_i, episode_ids[i])
                    if evidence_gate is not None and target is not None:
                        gate_verdicts[step, i] = evidence_gate.verdict(
                            teachers[i], u_i, advice)
                        if args.teacher == 'llm_general':
                            gate_label_ids[(step, i)] = (
                                f'{run_name}:{iteration}:{step}:{i}')
                    if target is None:
                        # An empty answer has two entirely different
                        # causes and merging them makes the teacher
                        # look broken. The advisor DECLINING (budget
                        # spent, gate closed) never reached the
                        # teacher and costs nothing; the teacher being
                        # ASKED and returning nothing usable is a real
                        # failure worth watching. A 32-consultation
                        # smoke reported 3032 "abstains" under the
                        # merged counter, which is the advisor
                        # declining after its budget ran out, not a
                        # 99% teacher failure rate.
                        if consulted:
                            total_abstains += 1
                        else:
                            total_declined += 1
                    else:
                        teacher_targets[step, i] = torch.from_numpy(
                            target
                        )
                        teacher_mask[step, i] = 1.0

                        # Explanation target for R2/R3/R4. Recorded
                        if consequence_data is not None:
                            consequence_data.collect(
                                u_i, student_action_i, target,
                                s3r3_phase(u_i), iteration-1, step, i)

                        # Existing text targets retain their own path.
                        # only where the teacher actually produced
                        # text: an action label without an explanation
                        # is a valid action sample and an invalid
                        # auxiliary one, which is why these masks are
                        # separate. The phase is read from the symbolic
                        # state, never from the text, so the R3
                        # grouping cannot inherit the content it is
                        # meant to corrupt.
                        # Collected whenever a teacher spoke, not only
                        # when a head exists. R1 pays for exactly the
                        # same text and must be auditable against R2 --
                        # otherwise "R1 collects it and discards it" is
                        # an assertion with no record behind it. R1 is
                        # never embedded, so auditing it is free.
                        if args.guidance and not args.offline_summary_only:
                            text = (advice.explanation or '').strip()
                            if text:
                                pending_expl.append({
                                    'step': step, 'env': i, 'text': text,
                                    'phase': s3r3_phase(u_i),
                                    'subgoal': advice.cost.metadata.get(
                                        'subgoal'),
                                    # Where this consultation happened,
                                    # zero-based, so recorded positions
                                    # can be checked against the
                                    # resolved schedule after the run.
                                    'rollout': iteration - 1,
                                    'episode': episode_ids[i],
                                    'global_transition': (
                                        (iteration - 1) * args.batch_size
                                        + step * args.num_envs + i
                                    ),
                                    'queried_at_utc': time.strftime(
                                        '%Y-%m-%dT%H:%M:%SZ',
                                        time.gmtime()
                                    ),
                                    # UNIQUE per transition. A text
                                    # hash is not an id: the same
                                    # explanation recurs on distinct
                                    # transitions, and collapsing them
                                    # made the shuffler hand a row its
                                    # own target back. The text hash is
                                    # retained separately, for embedding
                                    # cache identity only.
                                    'sample_id': (
                                        f'{run_name}:{iteration}:'
                                        f'{step}:{i}'
                                    ),
                                    'text_key':
                                        EmbeddingProvider.key_for(text),
                                })
                                if args.audit_explanations:
                                    # Capture BEFORE env.step and before
                                    # PPO changes the feature extractor.
                                    pending_expl[-1]['audit'] = {
                                        'features': _rollout_hidden[i]
                                        .detach().cpu().tolist(),
                                        'feature_stage': 'rollout_pre_update',
                                        'local_obs': obs[step, i]
                                        .detach().cpu().tolist(),
                                        'agent_pos': list(map(
                                            int, u_i.agent_pos)),
                                        'agent_dir': int(u_i.agent_dir),
                                        'mission': u_i.mission,
                                        'full_grid': u_i.grid.encode()
                                        .tolist(),
                                        'carrying': (
                                            None if u_i.carrying is None
                                            else [u_i.carrying.type,
                                                  u_i.carrying.color]),
                                        'student_action': student_action_i,
                                        'executed_action': student_action_i,
                                        'teacher_action': advice.action,
                                        'teacher_target': target.tolist(),
                                        'executed_since_query': list(
                                            ctx.get('executed_actions', [])),
                                    }
                                    # Preserve paid text/features even if
                                    # embedding at rollout end fails.
                                    _append_explanation_records(
                                        explanation_dir, iteration,
                                        [pending_expl[-1]], args,
                                        'collected_explanations.jsonl')
                        # Retain the label so it can go on
                        # contributing after this rollout is
                        # discarded. This is what makes a budget buy
                        # training signal rather than a single
                        # intervention.
                        if advice_replay is not None:
                            advice_replay.add(
                                obs[step, i].cpu().numpy(), target
                            )
            elif (
                coef > 0.0
                and not under_budget
                and not budget_notice_printed
            ):
                # The cap just took effect this step -- report it
                # once, then go quiet; teacher/budget_exhausted in
                # the per-iteration diagnostics stays visible for
                # the rest of the run.
                print(
                    f'[budget] teacher spend reached '
                    f'${total_cost.dollars:.4f} >= '
                    f'max_cost_dollars={args.max_cost_dollars}; '
                    f'no further teacher queries this run -- '
                    f'training continues as pure PPO.'
                )
                budget_notice_printed = True

            executed = action.cpu().numpy()
            if execute_p > 0.0:
                # RLingua-style (addendum 8): where this step carries a
                # teacher label, execute the teacher's action with
                # probability execute_p. The stored action and its
                # log-probability become the executed one's under the
                # student's own policy, so PPO's update sees what
                # actually happened in the environment.
                overridden = False
                for i in range(args.num_envs):
                    if (teacher_mask[step, i] > 0
                            and execute_rng.random() < execute_p):
                        executed[i] = int(teacher_targets[step, i].argmax())
                        overridden = True
                        teacher_executed += 1
                if overridden:
                    executed_t = torch.as_tensor(
                        executed, dtype=actions.dtype, device=device)
                    actions[step] = executed_t
                    logprobs[step] = torch.log_softmax(
                        logits, dim=-1).gather(
                            1, executed_t.long().view(-1, 1)).view(-1)
            if rlingua is not None:
                # RLingua Algorithm A-1, line 4: the controller acts
                # instead of the student with probability p, which decays
                # by rlingua_decay per environment transition (line 6).
                # Autoreset ticks (dones[step]) are skipped: the env
                # discards that action, so it is neither the controller's
                # nor anything to clone.
                rl_exec[step] = 0.0
                p_rl = args.rlingua_p0 * args.rlingua_decay ** rl_transitions
                starts_np = episode_starts[step].cpu().numpy()
                reset_np = dones[step].cpu().numpy()
                rl_transitions += int((reset_np == 0).sum())
                taken = False
                for i in range(args.num_envs):
                    if reset_np[i] > 0:
                        continue
                    if starts_np[i] > 0:
                        rlingua.reset(i)        # fresh instance per episode
                    act_now = rl_rng.random() < p_rl
                    if not (act_now or rlingua.every_step):
                        continue
                    a_ctrl = rlingua.act(i, sync_envs.envs[i].unwrapped)
                    if not act_now:
                        continue
                    executed[i] = a_ctrl
                    rl_exec[step, i] = 1.0
                    taken = True
                    rl_executed += 1
                    j = rl_buf_next
                    rl_buf_obs[j] = obs[step, i].to(torch.uint8).cpu()
                    if rl_buf_core is not None:
                        rl_buf_core[j] = core_in[0, i].cpu()
                    rl_buf_start[j] = float(starts_np[i])
                    rl_buf_act[j] = int(a_ctrl)
                    rl_buf_next = (j + 1) % rl_cap
                    rl_buf_n = min(rl_buf_n + 1, rl_cap)
                for i in range(args.num_envs):
                    if reset_np[i] == 0:        # what the env will run
                        rlingua.executed(i, executed[i])
                if taken:
                    actions[step] = torch.as_tensor(
                        executed, dtype=actions.dtype, device=device)
            next_obs, reward, terminations, truncations, infos = (
                envs.step(executed)
            )
            done_mask = np.logical_or(terminations, truncations)
            if coverage is not None:
                coverage.step(dones[step].cpu().numpy(), done_mask)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            if auxiliary is not None and hasattr(auxiliary, 'shape_reward'):
                # Opt-in training-only shaping (algos.ppo_potential).
                # Episode statistics and teacher-off evaluation keep the
                # unshaped task reward.
                rewards[step] += torch.as_tensor(
                    auxiliary.shape_reward(
                        sync_envs, dones[step].cpu().numpy().astype(bool),
                        done_mask, global_step),
                    dtype=rewards.dtype, device=device)
            next_obs = torch.Tensor(next_obs).to(device)
            if next_obs_buf is not None:
                next_obs_buf[step] = (
                    next_obs if args.count_observation == 'policy' else
                    local_count_observations(sync_envs, device))
            next_done = torch.Tensor(done_mask).to(device)

            # Keep the stateful bot synchronized: remember what the
            # student actually did, and flag envs whose episode just
            # ended so the next query rebuilds the plan for the new
            # mission (the env auto-resets on the following step).
            # dones[step] flags envs whose PREVIOUS episode ended, so
            # this tick is Gymnasium's NEXT_STEP autoreset: the action
            # submitted for it is discarded by the environment and must
            # not be reported to the teacher as something the student
            # did.
            reset_only = dones[step].cpu().numpy().astype(bool)
            for i in range(args.num_envs):
                last_actions[i] = int(executed[i])
                note_executed_action(
                    executed_since_query[i], executed[i], reset_only[i]
                )
                if done_mask[i]:
                    fresh_episode[i] = True
                    reference_fresh[i] = True
                    episode_ids[i] += 1
                    last_actions[i] = None
                    # A new episode resets the teacher's history, so
                    # nothing from the finished one may carry over.
                    executed_since_query[i].clear()

            if 'episode' in infos:
                mask = infos['_episode']
                for i in range(args.num_envs):
                    if not mask[i]:
                        continue
                    ep_return = float(infos['episode']['r'][i])
                    ep_length = float(infos['episode']['l'][i])
                    writer.add_scalar(
                        'charts/episodic_return',
                        ep_return,
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_length',
                        ep_length,
                        global_step,
                    )
                    writer.add_scalar(
                        'charts/episodic_success',
                        1.0 if ep_return > 0 else 0.0,
                        global_step,
                    )
                    recent_returns.append(ep_return)
                    recent_successes.append(
                        1.0 if ep_return > 0 else 0.0
                    )
                    tracker.record_episode(
                        global_step, i, ep_return, ep_length
                    )

        # Count the declared observation partition in each logged arm.
        # The no-bonus control records coverage without using it in
        # either the policy advantage or the value loss.
        curiosity = torch.zeros_like(rewards)
        exploration_stats = {}
        if exploration is not None:
            curiosity, exploration_stats = exploration.rollout(
                obs, next_obs_buf, actions, dones)
        if coverage is not None:
            exploration_stats.update(coverage.stats())

        # --- Advantage estimation (GAE), identical to ppo.py ---
        with torch.no_grad():
            next_values = agent.get_value(
                next_obs,
                core_state=core_state,
                episode_start=episode_start,
            )
            next_value = next_values[:, 0].reshape(1, -1)
            # RLingua: the student's traces stop where the controller
            # acted, so no student advantage or value target contains
            # the controller's actions or rewards (RLingua's critic
            # learns from R_RL only; TD3 bootstraps with the actor's own
            # next action). None everywhere else: unchanged arithmetic.
            student = None if rlingua is None else 1.0 - rl_exec
            advantages = extrinsic_gae(
                rewards, values, dones, next_value, next_done,
                args.gamma, args.gae_lambda, student)
            returns = advantages + values
            if args.bonus != 'none':
                int_adv, int_returns = intrinsic_gae(
                    curiosity, int_values, next_values[:, 1],
                    args.int_gamma, args.gae_lambda, keep=student)
                advantages = advantages + args.int_coef * int_adv
                b_int_returns = int_returns.reshape(-1)

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if evidence_gate is not None:
            usable = teacher_mask.detach().cpu().numpy() > 0
            keep, gate_row = evidence_gate.finalize(
                usable, gate_verdicts, coef > 0.0)
            teacher_mask.copy_(torch.as_tensor(
                keep, dtype=teacher_mask.dtype, device=teacher_mask.device))
            with open(gate_log_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'rollout': iteration - 1, 'global_step': global_step,
                    'coef': float(coef), **gate_row}) + '\n')
            if gate_label_ids:
                with open(gate_label_path, 'a', encoding='utf-8') as handle:
                    for (s_i, e_i), sample_id in sorted(
                            gate_label_ids.items()):
                        handle.write(json.dumps({
                            'sample_id': sample_id,
                            'verdict': bool(gate_verdicts[s_i, e_i]),
                            'accepted': bool(keep[s_i, e_i]),
                            'used_for_training': bool(
                                keep[s_i, e_i] and coef > 0.0),
                        }) + '\n')
            gate_verdicts[:] = False
            gate_label_ids.clear()
        if rule_timing is not None:
            rule_timing.finalize(
                teacher_mask.detach().cpu().numpy(), iteration - 1,
                global_step, coef)
        timing_minibatches = timing_labeled_minibatches = 0
        timing_loss_sum = 0.0
        b_teacher_targets = teacher_targets.reshape(-1, n_actions)
        b_teacher_mask = teacher_mask.reshape(-1)
        # RLingua: 0 where the controller acted (its R_LLM data) and on
        # Gymnasium's autoreset ticks (dones: the env discarded that
        # action), so the RL terms below use only the student's own real
        # transitions. (The other arms keep the trainer's original
        # convention, which includes autoreset ticks; excluding them here
        # can only help RLingua.)
        b_student = ((1.0 - rl_exec) * (1.0 - dones)).reshape(-1)
        # Resolve this rollout's explanations into frozen embedding
        # targets. R3 draws its donors here, from the rollout's own
        # eligible pool, so a donor is always a real explanation that
        # actually occurred in the same phase.
        if pending_expl and explanation_head is not None:
            texts = [row['text'] for row in pending_expl]
            if args.explanation_target == 'subgoal':
                from teachers.controlled_advice import subgoal_vectors
                vectors = subgoal_vectors(pending_expl)
            else:
                vectors = embedder.embed(texts)
            # Eligibility is computed identically in R2, R3 and R4:
            # a sample survives only if a within-phase donor existed
            # for it. Only R3 then swaps in the donor's target.
            donors, usable_arr = shuffler.donor_indices(
                [row['phase'] for row in pending_expl],
                valid=[True] * len(pending_expl),
                sample_ids=[row['sample_id'] for row in pending_expl],
                targets=([row['subgoal'] for row in pending_expl]
                         if args.explanation_target == 'subgoal' else texts),
            )
            usable = list(usable_arr)
            if args.explanation == 'shuffled':
                vectors = [vectors[d] for d in donors]
            for row, donor in zip(pending_expl, donors):
                row['donor_sample_id'] = pending_expl[donor]['sample_id']
                # Only R3 trains on the donor. R2 and R4 keep their own
                # target while running the identical eligibility rule.
                row['target_sample_id'] = (
                    row['donor_sample_id']
                    if args.explanation == 'shuffled'
                    else row['sample_id']
                )

            for row, vector, ok in zip(pending_expl, vectors, usable):
                row['usable'] = bool(ok)
                if not ok:
                    continue
                expl_targets[row['step'], row['env']] = torch.as_tensor(
                    vector, dtype=torch.float32, device=device
                )
                expl_mask[row['step'], row['env']] = 1.0

            if embedder is not None:
                embedder.save()

        # Persist the evidence incrementally, for EVERY arm that has a
        # teacher. Without this the claimed unchanged-target report and
        # the reusable cache do not exist after a real run: pending_expl
        # is cleared each rollout and only policy weights are saved.
        # Written per rollout so an interrupted run keeps what it
        # collected. R1 reaches here with no embedding and no donor,
        # which is the point: its text is audited, never trained on.
        if pending_expl:
            _append_explanation_records(
                explanation_dir, iteration, pending_expl, args
            )
            expl_records_written += len(pending_expl)
            queried_rollouts.add(iteration - 1)

        b_expl_targets = expl_targets.reshape(-1, args.embed_dim)
        b_expl_mask = expl_mask.reshape(-1)
        if consequence_data is not None:
            b_effect_actions, b_effect_targets, b_effect_mask = (
                consequence_data.resolve(iteration-1))
        b_episode_starts = episode_starts.reshape(-1)

        # flat_grid[t, e] locates obs[t, e] in the flat arrays, which
        # is how a recurrent minibatch selects every timestep of a
        # chosen group of envs in the right (T-major) order.
        flat_grid = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )

        # --- PPO update + the distillation term ---
        b_inds = np.arange(args.batch_size)
        env_inds = np.arange(args.num_envs)
        clipfracs = []
        if auxiliary is not None:
            auxiliary.begin_rollout(iteration)
        for epoch in range(args.update_epochs):
            # A recurrent core needs temporally contiguous data, so
            # minibatches are whole ENVS there. The feed-forward path
            # keeps the original flat timestep shuffle, so existing
            # runs are bit-for-bit unaffected.
            if args.recurrent:
                np.random.shuffle(env_inds)
                batches = [
                    (
                        flat_grid[
                            :, env_inds[s:s + envs_per_batch]
                        ].ravel(),
                        env_inds[s:s + envs_per_batch],
                    )
                    for s in range(
                        0, args.num_envs, envs_per_batch
                    )
                ]
            else:
                np.random.shuffle(b_inds)
                batches = [
                    (b_inds[s:s + args.minibatch_size], None)
                    for s in range(
                        0, args.batch_size, args.minibatch_size
                    )
                ]

            for mb_inds, mb_envs in batches:
                # Replay starts from the state this iteration's
                # rollout started from, for these envs specifically.
                mb_core = (
                    initial_core_state[:, mb_envs]
                    if args.recurrent
                    else None
                )
                mb_starts = (
                    b_episode_starts[mb_inds]
                    if args.recurrent
                    else None
                )

                (
                    _, newlogprob, entropy, newvalue, logits, _,
                    newhidden,
                ) = (
                    agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        core_state=mb_core,
                        episode_start=mb_starts,
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                # Same NaN guard as the baseline. Note the ratio
                # here is always well-behaved in principle (the
                # actions really were sampled from pi_old), so this
                # is belt-and-braces, not a load-bearing fix as it
                # was for override.
                logratio = torch.clamp(logratio, -10.0, 10.0)
                ratio = logratio.exp()

                mb_student = b_student[mb_inds] if rlingua is not None \
                    else None
                with torch.no_grad():
                    approx_kl = student_mean(
                        (ratio - 1) - logratio, mb_student)
                    clipfracs += [
                        student_mean(
                            ((ratio - 1.0).abs() > args.clip_coef).float(),
                            mb_student)
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv and mb_student is not None:
                    # Normalize over the student's own steps only: the
                    # controller's steps carry no policy-gradient term.
                    n_st = mb_student.sum().clamp(min=1.0)
                    m_st = (mb_advantages * mb_student).sum() / n_st
                    s_st = (((mb_advantages - m_st) ** 2 * mb_student).sum()
                            / n_st).sqrt()
                    mb_advantages = (mb_advantages - m_st) / (s_st + 1e-8)
                elif args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_per_sample = torch.max(pg_loss1, pg_loss2)
                advisor_weight = aux_loss_advisor = None
                if aux_actor is not None:
                    # ADVISOR: the auxiliary actor reads the same
                    # features and learns the teacher by imitation only;
                    # its fit estimates the imitation gap per state.
                    adv_mask = b_teacher_mask[mb_inds]
                    aux_log = F.log_softmax(aux_actor(newhidden), dim=-1)
                    aux_ce = -(b_teacher_targets[mb_inds] * aux_log).sum(-1)
                    with torch.no_grad():
                        advisor_weight = torch.exp(
                            -args.advisor_alpha * aux_ce) * adv_mask
                        if adv_mask.sum() > 0:
                            advisor_stats['weight_sum'] += float(
                                advisor_weight.sum())
                            advisor_stats['aux_ce_sum'] += float(
                                (aux_ce * adv_mask).sum())
                            advisor_stats['labels'] += int(adv_mask.sum())
                    if coef > 0.0 and adv_mask.sum() > 0:
                        aux_loss_advisor = masked_distillation_loss(
                            aux_ce, adv_mask, args.distill_normalization)
                    # (1 - w) on the RL loss while the teacher is active;
                    # the weight fades with the annealed coefficient so
                    # the teacher-free quarter is plain PPO, as in every
                    # other arm.
                    pg_loss = ((1.0 - min(1.0, coef) * advisor_weight)
                               * pg_per_sample).mean()
                elif mb_student is not None:
                    # RLingua: the RL actor term uses R_RL only.
                    pg_loss = ((pg_per_sample * mb_student).sum()
                               / mb_student.sum().clamp(min=1.0))
                else:
                    pg_loss = pg_per_sample.mean()

                new_int_value = newvalue[:, 1] if args.dual_value else None
                newvalue = newvalue[:, 0]
                if args.clip_vloss:
                    v_loss_unclipped = (
                        newvalue - b_returns[mb_inds]
                    ) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (
                        v_clipped - b_returns[mb_inds]
                    ) ** 2
                    v_loss_max = torch.max(
                        v_loss_unclipped, v_loss_clipped
                    )
                    v_loss = 0.5 * student_mean(v_loss_max, mb_student)
                else:
                    v_loss = 0.5 * student_mean(
                        (newvalue - b_returns[mb_inds]) ** 2, mb_student)

                # Match ppo_intrinsic's unclipped intrinsic critic.
                # A dormant head in the no-bonus arm contributes no
                # gradients, so that control still optimizes task PPO.
                int_v_loss = torch.zeros((), device=device)
                if args.bonus != 'none':
                    int_v_loss = 0.5 * student_mean((
                        new_int_value - b_int_returns[mb_inds]
                    ).square(), mb_student)

                # The kickstarting term: cross-entropy from the
                # (fixed) teacher target to the student's policy,
                # averaged over the states that have a target.
                # Equals KL(q || pi) up to a theta-free constant;
                # for one-hot q it is exactly -log pi(a*|s).
                mb_mask = b_teacher_mask[mb_inds]
                if coef > 0.0 and mb_mask.sum() > 0:
                    log_pi = F.log_softmax(logits, dim=-1)
                    ce = -(
                        b_teacher_targets[mb_inds] * log_pi
                    ).sum(dim=-1)
                    if advisor_weight is not None:
                        ce = advisor_weight * ce
                    distill_loss = masked_distillation_loss(
                        ce, mb_mask, args.distill_normalization)
                else:
                    distill_loss = torch.zeros(
                        (), device=device
                    )
                if rule_timing is not None:
                    timing_minibatches += 1
                    timing_labeled_minibatches += int(
                        coef > 0.0 and mb_mask.sum().item() > 0)
                    timing_loss_sum += float(distill_loss.detach().item())

                # Replay term: the same cross-entropy, but over
                # labels bought in EARLIER rollouts. Without it a
                # label contributes for one rollout and is thrown
                # away, so a budget of n buys n * update_epochs
                # gradient steps and no more. With it, n labels
                # keep contributing for the rest of training, which
                # is what makes the budget question here an active
                # learning question rather than a scheduling one.
                if (
                    advice_replay is not None
                    and coef > 0.0
                    and len(advice_replay) > 0
                ):
                    replay_batch = (
                        args.advice_replay_batch
                        or args.minibatch_size
                    )
                    sampled = advice_replay.sample(replay_batch)
                    if sampled is not None:
                        r_obs, r_targets = sampled
                        r_obs_t = torch.as_tensor(
                            r_obs, device=device, dtype=torch.float32
                        )
                        r_targets_t = torch.as_tensor(
                            r_targets, device=device
                        )
                        # Scored feed-forward: replayed states are
                        # sampled out of order, so a recurrent core
                        # has no valid hidden state for them. The
                        # encoder and heads are shared, so the
                        # cloning signal still lands on the weights
                        # that matter.
                        r_logits = agent.actor(
                            agent._encode(r_obs_t)
                        )
                        r_log_pi = F.log_softmax(r_logits, dim=-1)
                        replay_loss = -(
                            r_targets_t * r_log_pi
                        ).sum(dim=-1).mean()
                        distill_loss = (
                            distill_loss
                            + args.advice_replay_coef * replay_loss
                        )

                entropy_loss = student_mean(entropy, mb_student)
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + (v_loss + int_v_loss) * args.vf_coef
                    + coef * distill_loss
                )
                if (rlingua is not None and rl_buf_n > 0
                        and args.rlingua_bc_coef > 0.0):
                    # RLingua Algorithm A-1, line 10: behavior cloning on
                    # a batch drawn from the persistent controller buffer
                    # (R_LLM), as large as the student minibatch, at
                    # constant weight lambda_IM. Cross-entropy is the
                    # discrete-action counterpart of its squared error.
                    # Each stored step is replayed from the recurrent
                    # state it was taken in (stored-state replay).
                    bc_idx = torch.as_tensor(rl_bc_rng.integers(
                        0, rl_buf_n, size=len(mb_inds)))
                    bc_core = (rl_buf_core[bc_idx].to(device).unsqueeze(0)
                               if rl_buf_core is not None else None)
                    bc_hidden, _ = agent.get_states(
                        rl_buf_obs[bc_idx].to(device).float(),
                        bc_core, rl_buf_start[bc_idx].to(device))
                    bc_loss = F.cross_entropy(
                        agent.actor(bc_hidden), rl_buf_act[bc_idx].to(device))
                    loss = loss + args.rlingua_bc_coef * bc_loss
                    rl_bc_sum += float(bc_loss.detach().item())
                    rl_bc_count += 1

                # Auxiliary explanation loss (R2/R3/R4). Masked to the
                # transitions that carry a valid target; unqueried steps
                # still train PPO and contribute nothing here. R4
                # detaches, so its gradient stops before the trunk.
                if explanation_head is not None:
                    # newhidden is ALREADY in minibatch order: the
                    # forward above received b_obs[mb_inds]. Indexing it
                    # again with the global mb_inds crashes as soon as
                    # there is more than one minibatch, and silently
                    # misaligns features against targets even with one.
                    # The TARGETS are global and must still be indexed.
                    aux_pred = explanation_head(
                        newhidden,
                        detach_features=(args.explanation == 'detached'),
                    )
                    # Structured labels use categorical supervision;
                    # sentence targets retain their cosine objective.
                    from algos.explanation_head import subgoal_loss
                    target_loss = (subgoal_loss
                                   if args.explanation_target == 'subgoal'
                                   else explanation_loss)
                    aux_loss = target_loss(
                        aux_pred,
                        b_expl_targets[mb_inds],
                        b_expl_mask[mb_inds],
                    )
                    loss = loss + args.lambda_aux * aux_loss
                    if (args.audit_explanations
                            and bool(b_expl_mask[mb_inds].any())):
                        # Measure the gradient at the shared feature
                        # boundary without the unnecessary cost of an
                        # extra traversal through the recurrent core.
                        # Normal loss.backward below updates parameters.
                        gradients = torch.autograd.grad(
                            args.lambda_aux * aux_loss,
                            newhidden, retain_graph=True,
                            allow_unused=True)
                        norm_squared = sum(
                            float(g.detach().double().square().sum())
                            for g in gradients if g is not None)
                        with open(os.path.join(
                                explanation_dir, 'auxiliary_updates.jsonl'),
                                'a', encoding='utf-8') as handle:
                            handle.write(json.dumps({
                                'iteration': iteration,
                                'rollout': iteration - 1,
                                'global_step': global_step,
                                'epoch': epoch,
                                'valid_targets': int(
                                    b_expl_mask[mb_inds].sum().item()),
                                'cosine_loss': float(aux_loss.detach()),
                                'lambda_aux': args.lambda_aux,
                                'weighted_aux_hidden_grad_l2':
                                    math.sqrt(norm_squared),
                            }, allow_nan=False) + '\n')

                if (consequence_head is not None
                        and bool(b_effect_mask[mb_inds].any())):
                    from algos.consequence_head import consequence_loss
                    effect_mask = b_effect_mask[mb_inds]
                    effect_loss = consequence_loss(
                        consequence_head(
                            newhidden, b_effect_actions[mb_inds],
                            detach_features=(args.consequence == 'detached')),
                        b_effect_targets[mb_inds], effect_mask,
                        b_effect_actions[mb_inds])
                    # Fix each target's epoch weight independently of
                    # uneven eligible counts in recurrent minibatches.
                    eligible_mb = effect_mask.any(-1).any(-1).sum()
                    eligible_all = b_effect_mask.any(-1).any(-1).sum()
                    effect_loss = effect_loss * (
                        args.num_minibatches*eligible_mb/eligible_all)
                    if bool(effect_mask.any()) and epoch == 0:
                        grad = torch.autograd.grad(
                            args.consequence_coef*effect_loss, newhidden,
                            retain_graph=True, allow_unused=True)[0]
                        evidence = dict(
                            rollout=iteration-1, epoch=epoch,
                            samples=int(effect_mask.any(-1).any(-1).sum()),
                            loss=float(effect_loss.detach()),
                            hidden_grad_l2=(float(grad.norm())
                                            if grad is not None else 0.0))
                        from algos.consequence_training import (
                            gradient_diagnostics)
                        evidence.update(gradient_diagnostics(
                            args.consequence_coef*effect_loss,
                            loss-coef*distill_loss, coef*distill_loss, agent))
                        evidence['sample_ids'] = [
                            f'{iteration-1}:{int(j)//args.num_envs}:'
                            f'{int(j)%args.num_envs}'
                            for j, active in zip(
                                mb_inds, effect_mask.any(-1).any(-1).tolist())
                            if active]
                        with (consequence_data.directory/'updates.jsonl').open(
                                'a') as handle:
                            handle.write(json.dumps(evidence)+'\n')
                    loss = loss + args.consequence_coef*effect_loss
                if auxiliary is not None:
                    # Pass the same sequence features used by the actor.
                    # Targets stay private to training, outside the policy.
                    if hasattr(auxiliary, 'set_batch'):
                        auxiliary.set_batch(
                            mb_inds, newhidden, b_obs[mb_inds]
                        )
                    loss = loss + auxiliary.loss()
                if aux_loss_advisor is not None:
                    # Shares the representation, as in ADVISOR.
                    loss = loss + aux_loss_advisor
                optimizer.zero_grad()
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad()
                loss.backward()
                # POLICY PARAMETERS ONLY, and this is load-bearing for
                # R4: including the explanation head here would let its
                # gradients enter the global norm and rescale the
                # policy's gradients by an amount that depends on the
                # auxiliary loss, so R4 would differ from R1 for a
                # reason unrelated to representation learning.
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                if explanation_head is not None:
                    nn.utils.clip_grad_norm_(
                        explanation_head.parameters(), args.max_grad_norm
                    )
                if consequence_head is not None:
                    nn.utils.clip_grad_norm_(
                        consequence_head.parameters(), args.max_grad_norm)
                optimizer.step()
                if aux_optimizer is not None:
                    nn.utils.clip_grad_norm_(
                        aux_actor.parameters(), args.max_grad_norm)
                    aux_optimizer.step()
                if auxiliary is not None:
                    auxiliary.optimizer_step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        if rule_timing is not None:
            rule_timing.optimization(
                iteration - 1, timing_labeled_minibatches,
                timing_minibatches, timing_loss_sum)
        if args.advisor_rng_isolation:
            # Count applied rollout labels before buffers are reset.
            # Sham slots never enter these masks or this label count.
            teacher_labels += int(teacher_mask.sum().item())
            if first_label_global_step is None and teacher_mask.any():
                first_step = int(torch.nonzero(teacher_mask)[0, 0])
                first_label_global_step = (
                    global_step - args.batch_size
                    + (first_step + 1) * args.num_envs)

        # Reset the target buffers for the next rollout so a step
        # that goes unqueried next iteration cannot inherit this
        # iteration's stale target.
        teacher_targets.zero_()
        teacher_mask.zero_()
        expl_targets.zero_()
        expl_mask.zero_()
        pending_expl.clear()

        # --- Teacher-off greedy evaluation ---
        # The internalization measurement: the student alone, argmax
        # actions, no teacher anywhere. Same fixed episode set every
        # time (see monitoring/eval.py).
        run_eval = args.eval_interval > 0 and (
            iteration % args.eval_interval == 0
            or iteration in eval_milestones
            or iteration == args.num_iterations
        )
        if run_eval:

            before_eval = (policy_sha256(agent)
                           if iteration in diagnostic_points else None)
            last_eval = evaluate_student()
            if args.eval_sampled:
                writer.add_scalar(
                    'charts/eval_sampled_success_rate',
                    last_eval['sampled_success_rate'],
                    global_step,
                )
            writer.add_scalar(
                'charts/eval_success_rate',
                last_eval['success_rate'],
                global_step,
            )
            writer.add_scalar(
                'charts/eval_return',
                last_eval['mean_return'],
                global_step,
            )
            writer.add_scalar(
                'charts/eval_length',
                last_eval['mean_length'],
                global_step,
            )
            # Persist actual evaluation points, not the last evaluation
            # repeated in every training summary. This supports an
            # auditable teacher-off learning curve after metadata rsync.
            with open(os.path.join(run_dir, 'evaluations.jsonl'), 'a',
                      encoding='utf-8') as handle:
                handle.write(json.dumps({
                    'global_step': global_step,
                    'iteration': iteration,
                    'teacher_on': False,
                    'episodes': args.eval_episodes,
                    'seed_base': args.seed + 50_000,
                    'training_episodes': tracker.episode_count,
                    'wall_time_sec': time.time() - start_time,
                    **last_eval,
                }, allow_nan=False) + '\n')

        if iteration in diagnostic_points:
            record_diagnostic(
                iteration, last_eval if run_eval else None,
                before_eval if run_eval else None)

        # --- Per-iteration diagnostics ---
        # RLingua: over the student's own real transitions only.
        keep_ev = (slice(None) if rlingua is None
                   else (b_student > 0).cpu().numpy())
        y_pred = b_values.cpu().numpy()[keep_ev]
        y_true = b_returns.cpu().numpy()[keep_ev]
        var_y = np.var(y_true)
        explained_var = (
            np.nan
            if var_y == 0
            else 1 - np.var(y_true - y_pred) / var_y
        )

        writer.add_scalar(
            'charts/learning_rate',
            optimizer.param_groups[0]['lr'],
            global_step,
        )
        writer.add_scalar(
            'losses/value_loss', v_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/policy_loss', pg_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/entropy', entropy_loss.item(), global_step
        )
        writer.add_scalar(
            'losses/approx_kl', approx_kl.item(), global_step
        )
        writer.add_scalar(
            'losses/clipfrac', np.mean(clipfracs), global_step
        )
        writer.add_scalar(
            'losses/explained_variance', explained_var, global_step
        )
        writer.add_scalar(
            'losses/distill_loss', distill_loss.item(), global_step
        )
        writer.add_scalar(
            'teacher/distill_coef', coef, global_step
        )
        writer.add_scalar(
            'teacher/total_queries', total_queries, global_step
        )
        writer.add_scalar(
            'teacher/total_abstains', total_abstains, global_step
        )
        writer.add_scalar(
            'teacher/total_declined', total_declined, global_step
        )
        writer.add_scalar(
            'teacher/execute_probability', execute_p, global_step
        )
        writer.add_scalar(
            'teacher/total_executed', teacher_executed, global_step
        )
        if rlingua is not None:
            writer.add_scalar('rlingua/p', args.rlingua_p0
                              * args.rlingua_decay ** rl_transitions,
                              global_step)
            writer.add_scalar('rlingua/total_executed', rl_executed,
                              global_step)
            writer.add_scalar('rlingua/buffer', rl_buf_n, global_step)
            if rl_bc_count:
                writer.add_scalar('rlingua/bc_loss',
                                  rl_bc_sum / rl_bc_count, global_step)
        if evidence_gate is not None:
            writer.add_scalar(
                'teacher/evidence_rejected',
                evidence_gate.totals['rejected'], global_step)
        writer.add_scalar(
            'teacher/cost_dollars', total_cost.dollars, global_step
        )
        writer.add_scalar(
            'teacher/budget_exhausted',
            1.0 if budget_notice_printed else 0.0,
            global_step,
        )

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar('charts/SPS', sps, global_step)
        writer.add_scalar(
            'charts/wall_time_sec', tracker.elapsed(), global_step
        )
        writer.add_scalar(
            'charts/total_episodes', tracker.episode_count, global_step
        )
        if recent_successes:
            writer.add_scalar(
                'charts/success_rate_recent_100',
                sum(recent_successes) / len(recent_successes),
                global_step,
            )
        extra = {
            'learning_rate': optimizer.param_groups[0]['lr'],
            'value_loss': v_loss.item(),
            'intrinsic_value_loss': int_v_loss.item(),
            'exploration': exploration_stats,
            'policy_loss': pg_loss.item(),
            'entropy': entropy_loss.item(),
            'approx_kl': approx_kl.item(),
            'clipfrac': float(np.mean(clipfracs)),
            'explained_variance': float(explained_var),
            'distill_loss': distill_loss.item(),
            'distill_coef': coef,
            'teacher_total_queries': int(total_queries),
            # Consulted, and returned nothing usable. A real failure.
            'teacher_total_abstains': int(total_abstains),
            # Never consulted: the advisor declined. Costs nothing and
            # says nothing about the teacher.
            'teacher_total_declined': int(total_declined),
            # Teacher actions executed in place of the student's
            # (addendum 8's RLingua-style arm; 0 everywhere else).
            'teacher_total_executed': int(teacher_executed),
            'teacher_execute_probability': float(execute_p),
            'rlingua': None if rlingua is None else {
                'controller_sha256': rlingua.sha256,
                'variant': args.rlingua_variant,
                'p': float(args.rlingua_p0
                           * args.rlingua_decay ** rl_transitions),
                'transitions': int(rl_transitions),
                'total_executed': int(rl_executed),
                'buffer': int(rl_buf_n),
                'bc_loss_mean': (rl_bc_sum / rl_bc_count
                                 if rl_bc_count else None),
                'controller_calls': int(rlingua.calls),
                'controller_errors': int(rlingua.errors),
                'controller_invalid': int(rlingua.invalid)},
            'teacher_total_aliased': int(total_aliased),
            'imitation_weighting': args.imitation_weighting,
            'advisor_mean_weight': (
                advisor_stats['weight_sum'] / advisor_stats['labels']
                if advisor_stats['labels'] else None),
            'advisor_mean_aux_ce': (
                advisor_stats['aux_ce_sum'] / advisor_stats['labels']
                if advisor_stats['labels'] else None),
            'teacher_total_gated': int(
                evidence_gate.totals['rejected']
                if evidence_gate is not None else 0),
            **(evidence_gate.stats() if evidence_gate is not None
               else {'evidence_gate': 'none'}),
            'teacher_cost_dollars': total_cost.dollars,
            'teacher_wall_time_s': total_cost.wall_time_s,
            'teacher_compute_units': int(total_cost.compute_units),
            # Where the budget went, and how much of a typical
            # episode the advisor was in a position to label.
            'advising': advisor.stats(),
            'reference_teacher': args.action_reference or (
                args.teacher if args.teacher_stream else None),
            'reference_calls': reference_calls,
            'reference_labels': reference_labels,
            'reference_wall_time_s': reference_cost.wall_time_s,
            'reference_compute_units': int(reference_cost.compute_units),
        }
        if args.advisor_rng_isolation:
            extra['advisor_control'] = {
                'rng_mode': 'isolated_numpy_query_stream_v1',
                'query_seed': args.seed + 90_117,
                'sham': args.advisor_sham,
                'teacher_instances': sum(t is not None for t in teachers),
                'query_order_draws': query_order_draws,
                'teacher_labels': teacher_labels,
                'first_label_global_step': first_label_global_step,
            }
        if args.guidance:
            # Component counters do not become a report by existing.
            # Everything a spend or control claim would need is written
            # here, per iteration, so an interrupted run still has it.
            explanation_stats = {
                'arm': args.explanation,
                'target_kind': args.explanation_target,
                'records_written': int(expl_records_written),
                'rollouts_with_queries': len(queried_rollouts),
            }
            if embedder is not None:
                explanation_stats['embedding'] = embedder.stats(
                    price_per_million=args.embed_price_per_million
                )
            if shuffler is not None:
                explanation_stats['shuffle_coverage'] = shuffler.coverage()
            extra['explanation'] = explanation_stats
            # `llm_subgoal` can reuse one paid sub-goal across several
            # free BFS action labels. Persist component counters so reports
            # distinguish API planning calls from consultations and labels.
            component_stats = [
                teacher.stats() for teacher in teachers
                if callable(getattr(teacher, 'stats', None))
            ]
            if component_stats:
                numeric = {
                    key: sum(int(row.get(key, 0)) for row in component_stats)
                    for key in ('num_api_calls', 'num_replans', 'num_failures')
                    if any(key in row for row in component_stats)
                }
                extra['teacher_components'] = {
                    'instances': len(component_stats), **numeric}
        if clock is not None:
            extra['clock_query_schedule'] = clock.manifest()
        if online_rules is not None:
            extra['online_rules'] = online_rules.summary()
        if uniform_queries is not None:
            extra['uniform_query_schedule'] = uniform_queries.manifest()
            if args.advisor_sham:
                extra['uniform_query_schedule']['sham_observed'] = list(
                    sham_uniform_slots)
        if windows is not None:
            extra['query_schedule'] = windows.manifest()
            # The positions actually used, against the positions
            # specified. A reviewer compares these two lists; a
            # rollout here that is absent from the schedule is a bug,
            # not a judgement call.
            extra['query_rollouts_observed'] = sorted(queried_rollouts)
            extra['query_rollouts_off_schedule'] = sorted(
                queried_rollouts - set(windows.rollouts)
            )
        for key, metric in exploration_stats.items():
            writer.add_scalar(f'exploration/{key}', metric, global_step)
        if advice_replay is not None:
            extra['advice_replay'] = advice_replay.stats()
        if last_eval is not None:
            extra['eval_success_rate'] = last_eval['success_rate']
            extra['eval_mean_return'] = last_eval['mean_return']
            # Record the sampled rate too, or the run reports a bare
            # 0.00 with no way to tell a student that internalized
            # nothing from one whose argmax merely cycles.
            if 'sampled_success_rate' in last_eval:
                extra['eval_sampled_success_rate'] = last_eval[
                    'sampled_success_rate'
                ]
        tracker.write_summary(
            status='running', global_step=global_step,
            extra={**extra, 'consultations': consultation_journal.stats()}
        )

        if recent_returns:
            avg_succ = sum(recent_successes) / len(recent_successes)
            avg_ret = sum(recent_returns) / len(recent_returns)
            progress = f'success {avg_succ:4.2f} return {avg_ret:6.3f}'
        else:
            progress = 'success  --  return    --'
        eval_str = (
            f' | eval {last_eval["success_rate"]:4.2f}'
            if last_eval is not None
            else ''
        )
        print(
            f'iter {iteration}/{args.num_iterations} '
            f'step {global_step} SPS {sps} | {progress} '
            f'| ks {coef:5.2f}{eval_str}'
        )

    if budget is not None:
        embedding_spend = 0.0
        embedding_uncertain = False
        if embedder is not None:
            reported = embedder.stats(
                price_per_million=args.embed_price_per_million
            ).get('dollars')
            if reported is None:
                # Unknown usage is not zero usage. Settle at the amount
                # reserved for embeddings instead, so an unmeasurable
                # cost leaves the pool understating what is available
                # rather than overstating it.
                embedding_spend = budget_manifest['embedding_reserved_usd']
                embedding_uncertain = True
            else:
                embedding_spend = reported
        actual = float(total_cost.dollars) + float(embedding_spend)
        # A successful retry does not disclose billing for earlier timed
        # out attempts. Retain the full hold until the provider dashboard
        # is reconciled; measured response usage alone cannot release it.
        budget_manifest['settlement_pending'] = True
        budget_manifest['actual_teacher_usd'] = float(total_cost.dollars)
        budget_manifest['actual_embedding_usd'] = float(embedding_spend)
        budget_manifest['embedding_cost_uncertain'] = embedding_uncertain
        budget_manifest['actual_usd'] = actual
        # A run stopped by the cap is not a completed, matched arm, and
        # the manifest has to say so rather than leaving a reader to
        # infer it from a short query count.
        budget_manifest['budget_stop'] = bool(
            args.max_cost_dollars > 0
            and total_cost.dollars >= args.max_cost_dollars
        )
        extra['budget'] = budget_manifest
        print(f'budget: measured ${actual:.4f}; retaining '
              f'${budget_manifest["reserved_usd"]:.2f} reservation '
              f'for provider reconciliation')
    elif budget_manifest is not None:
        extra['budget'] = budget_manifest

    extra['consultations'] = consultation_journal.stats()
    tracker.close(
        status=('diagnostic_truncated' if stop_iteration < args.num_iterations
                else 'completed'), global_step=global_step, extra=extra)
    # Save the trained policy weights next to the run summary so the
    # sub-goal probe (scripts/probe_subgoal.py) can load this student
    # and test whether the teacher's sub-goal is decodable from its
    # internal representation -- the internalization measurement.
    torch.save(agent.state_dict(), os.path.join(run_dir, 'agent.pt'))
    if auxiliary is not None:
        auxiliary.finish(global_step)
    if consequence_head is not None:
        torch.save(consequence_head.state_dict(),
                   os.path.join(run_dir, 'consequences', 'head.pt'))
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
