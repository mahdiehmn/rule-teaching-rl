"""
PPO guided by a teacher via action override.

The method under test. A teacher proposes actions, and during a
fraction of steps -- annealed down over training -- the agent executes
the teacher's action instead of its own. PPO then learns from the
resulting trajectories: because the override produces successful
episodes the agent could not yet reach on its own, the advantage
signal pushes the policy toward the teacher's behavior, and once the
override anneals away the agent keeps acting on what it learned.

This file is a diff against algos/ppo.py. The only additions are:
  1. A teacher (here the free, perfect BFS oracle) queried from the
     underlying symbolic state while the agent still sees pixels.
  2. An ActionOverrider that decides, per env per step, whether to
     swap in the teacher's action, with the executed action's log-
     prob recomputed so PPO's importance ratio stays correct.
  3. Logging of the override rate and teacher query count / cost.

The teacher is pluggable: oracle (BFS, DoorKey), the free BabyAI
Bot ('bot', any BabyAI-registered task), or a VLM/LLM -- all
produce the same Advice object the overrider consumes.

The prefix / JSRL hand-off (--handoff prefix) with the bot teacher
is the fix for KeyCorridor, where loss-only distillation stalls at
0% because the student never completes an episode to earn reward;
the prefix hand-off has the teacher drive the opening so episodes
complete and PPO gets a reward signal to bootstrap from. See
docs/keycorridor_distill_diagnosis.md.

Run it:

    python -m algos.ppo_teacher --task doorkey_5x5           # oracle
    python -m algos.ppo_teacher --task keycorridor_s3r3 \\
        --teacher bot --handoff prefix                       # the fix
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium.wrappers.vector import RecordEpisodeStatistics
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from algos.nets import ObsEncoder, RecurrentCore, is_symbolic_mode
from envs.registry import TASKS, build_env, make_thunk
from advising import PeekableTeacher, make_advisor
from advising.policy import (
    logit_gap_importance,
    max_prob_importance,
    normalized_entropy_importance,
    policy_entropy_importance,
    probability_mistake,
    top2_gap_importance,
)
from envs.state import extract_doorkey_state
from integration.override import ActionOverrider, PrefixOverrider
from monitoring.eval import greedy_eval
from monitoring.metrics import RunTracker, runtime_metadata
from teachers.base import Cost
from teachers.minigrid.bfs_solver import MiniGridBFSTeacher


@dataclass
class Args:
    """
    Command-line arguments for a teacher-guided PPO run.

    Shares the PPO knobs with algos/ppo.py and adds the teacher /
    override settings at the bottom. tyro exposes each as a flag.
    """

    # --- Experiment identity and logging ---
    task: str = 'doorkey_5x5'
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = 'vlm-rl-bench'

    # --- Observation settings ---
    # 'historical' = the accumulated fog-of-war map (what the agent
    # has seen so far); 'partial' = egocentric agent view only;
    # 'full' / 'fully_obs' = the whole map. The teacher can still use
    # symbolic state or its own rendered image; this controls only
    # what the PPO agent sees. 'historical' is the project-wide
    # default every reported result uses -- keep it in sync with the
    # other algos and with scripts/submit_cc.sh.
    obs_mode: str = 'historical'
    # MiniGrid partial-view size. Must be odd and >= 3.
    agent_view_size: int = 7
    # RGB pixels per grid cell. 0 means auto-scale near
    # obs_target_size so the CNN input stays around 56x56.
    obs_tile_size: int = 0
    obs_target_size: int = 56
    # Add a GRU core between the trunk and the heads, so an override
    # arm can be measured at the same configuration as the
    # exploration-bonus arms instead of only on RGB, where nothing
    # reaches 0.20 and the comparison carries no information.
    recurrent: bool = False

    # --- Core PPO hyperparameters (same meaning as in ppo.py) ---
    total_timesteps: int = 1_000_000
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
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Teacher / override settings ---
    # Which teacher to consult: 'oracle' (free BFS planner, DoorKey
    # only), 'bot' (free BabyAI Bot, any BabyAI-registered task
    # including keycorridor_s3r3 / keycorridor_s6r3_babyai / gotoseq),
    # 'llm' (text description of the state), or 'vlm' (sees the
    # rendered full map). All produce the same Advice the overrider
    # consumes.
    teacher: str = 'oracle'
    # Optional override of the API teacher's model (e.g. 'gpt-4o').
    # Empty string uses each teacher's own default. Ignored by the
    # oracle.
    teacher_model: str = ''
    # Prompt variant for the LLM teacher ('default' or 'v2').
    teacher_prompt: str = 'default'
    # Per-env, per-step probability of executing the teacher's action
    # instead of the agent's, annealed start -> end.
    override_start: float = 0.75
    override_end: float = 0.0
    # Fraction of training over which the override probability decays.
    override_fraction: float = 0.5
    # Hand-off style. 'prob' = the original method: override RANDOM
    # steps with probability override_start->end. 'prefix' = Jump-
    # Start style: the teacher drives the first H steps of each
    # EPISODE and the agent owns the rest, with H annealed
    # handoff_start_len->handoff_end_len over override_fraction of
    # training. Prefix forces the agent to earn the task back to front
    # on its own policy, which transfers; prob masks the agent's
    # mistakes and tends to collapse when the override is removed.
    handoff: str = 'prob'
    # Guide prefix length H at the start / end of training (only used
    # when handoff='prefix'). start should cover the task's optimal
    # path so the teacher can solve a whole episode early on.
    handoff_start_len: int = 20
    handoff_end_len: int = 0

    # --- Advice budgeting (advising/) ---
    # The hand-off settings above decide WHICH steps the teacher is
    # allowed to drive. These decide how a limited quantity of advice
    # is SPENT across those steps, following Torrey & Taylor (AAMAS
    # 2013) adapted for a teacher that charges per query. The advisor
    # is a second gate inside the hand-off mask: a step is queried
    # only if the overrider flagged it AND the advisor approves.
    #
    # 'unlimited' (the default) approves everything, which is exactly
    # what this file did before budgeting existed, so every recorded
    # run reproduces unchanged.
    advisor: str = 'unlimited'
    # Cap on advice DELIVERED to the agent over the whole run
    # (Torrey & Taylor's n). 0 means unlimited.
    advice_budget: int = 0
    # Cap on teacher CONSULTATIONS over the whole run -- the dollar
    # budget for the LLM/VLM teachers. 0 means unlimited, which
    # reproduces the 2013 assumption that asking is free. Both
    # budgets are global across num_envs, not per-env.
    query_budget: int = 0
    # Importance threshold t, in [0, 1] against normalized
    # importance.
    importance_threshold: float = 0.0
    # How to score state importance. A PPO student has no Q-table,
    # so the paper's teacher-side measure is unavailable and the
    # substitutes read the policy's own uncertainty instead:
    #
    #   'entropy'      -- entropy / ln(num_actions). In [0, 1] in
    #                     every environment, so a threshold ports
    #                     across tasks. The default choice.
    #   'max_prob'     -- 1 - P(favourite action). Blunter; ignores
    #                     how the losing mass is arranged.
    #   'top2_gap'     -- 1 - (P(best) - P(second)). Highest when
    #                     the top two are neck-and-neck, which is
    #                     precisely when advice can still change
    #                     the behaviour. Closest in spirit to the
    #                     paper's max_a Q - min_a Q.
    #   'entropy_nats' -- raw entropy. Interpretable but its ceiling
    #                     is ln(num_actions), so a threshold does
    #                     NOT port between tasks with different
    #                     action counts.
    #   'logit_gap'    -- the literal translation of the paper's
    #                     formula to logits. Note the sign flips
    #                     meaning: this is the STUDENT's spread, so
    #                     high means confident, not important.
    #   'none'         -- importance disabled.
    importance_source: str = 'none'
    # By default I(s) is rescaled to [0, 1] from the running range
    # observed so far, so a threshold means roughly the same thing
    # across signals whose raw units differ. That rescaling is the
    # wrong choice for entropy, which already has an absolute and
    # interpretable scale: it runs from 0 (the policy is certain) to
    # ln(num_actions) (the policy is uniform), which for MiniGrid's
    # 7 actions is 1.946. Setting this reads
    # --importance-threshold in raw nats instead, where a value is
    # stable across training rather than drifting as the observed
    # range widens.
    raw_importance: bool = False
    # Target fraction of the steps the advisor sees that should
    # receive advice. Setting this above 0 replaces
    # --importance-threshold with a controller: the threshold is
    # nudged up when spending runs ahead of target and down when it
    # falls behind, so advice is spread across training instead of
    # being exhausted in the first few thousand steps.
    #
    # This is the recommended way to configure importance advising.
    # A fixed threshold is not portable -- the same number selects
    # 9.8% of states under 'max_prob' and 47.7% under 'top2_gap' on
    # the same run -- and it is not stationary either, because
    # policy uncertainty is high nearly everywhere early in training
    # and low nearly everywhere late. A rate is portable across both
    # signals and tasks, and says what you actually care about.
    #
    # 0 disables the controller and uses --importance-threshold as a
    # fixed cut.
    advice_rate: float = 0.0
    # How "the student is making a mistake" is decided, which the
    # mistake / predictive / surrogate advisors all depend on.
    #
    # 0.0 keeps Torrey & Taylor's test: the sampled action differs
    # from the teacher's. That test suits their epsilon-greedy
    # student, which has one intended action, but reads mostly noise
    # off a PPO student, which SAMPLES: a policy holding 90% of its
    # mass on the right action still draws something else one time
    # in ten, and correcting it there teaches nothing.
    #
    # Any value above 0 switches to the probability test -- a
    # mistake is pi(a_teacher | s) < this value, i.e. the student
    # does not actually believe in the teacher's action. 0.5 means
    # "the student is not already favouring the right move"; 0.1-0.2
    # reserves budget for states where it is badly wrong rather than
    # merely unsure. See advising/policy.probability_mistake.
    mistake_threshold: float = 0.0
    # Whether to wrap the teacher so its action can be looked up for
    # free (advising/peekable.py). This matters more than it sounds.
    #
    # Mistake correcting and predictive advising both need to know
    # the teacher's action BEFORE deciding whether to spend budget.
    # A Q-table teacher answers for free, which is why Torrey &
    # Taylor never have to think about it. An LLM does not, so
    # without a wrapper both algorithms collapse into plain
    # importance advising: they query at every state above the
    # threshold and use the answer only to decide whether to bother
    # the student. The money is spent regardless.
    #
    # 'none'      -- no wrapper. The teacher's own peek_action is
    #                used if it has one (the BFS oracle does), so
    #                this is the right setting for real runs with a
    #                free teacher.
    # 'blind'     -- ablation: wrap so that NO peek can ever be
    #                answered, turning a free teacher into a stand-in
    #                for a paid one. The baseline the other two are
    #                measured against, without spending on real API
    #                calls.
    # 'exact'     -- reuse answers already bought at the same state.
    #                Never wrong; in a gridworld, where states repeat
    #                constantly, this is most of the saving.
    # 'surrogate' -- exact recall plus a learned model of the
    #                teacher for unseen states. Saves more, and can
    #                be wrong.
    #
    # All three wrapper modes ignore the teacher's native
    # peek_action, so the arms face the same teacher and the
    # comparison is about what the wrapper contributes.
    peek: str = 'none'
    # Confidence the peek surrogate needs before its guess is used.
    peek_trust: float = 0.9
    # Action predictor for the predictive / surrogate advisors, and
    # for the peek wrapper.
    predictor: str = 'count'
    # Confidence the surrogate needs before its guess replaces a
    # real query.
    surrogate_trust: float = 0.8

    # --- Teacher-off evaluation (monitoring/eval.py) ---
    # This file previously ran NO evaluation at all: its only success
    # metric was `success_rate_recent_100`, the stochastic training
    # policy measured while the teacher was still overriding some of
    # its actions. That is not the quantity algos/ppo.py and
    # algos/ppo_distill.py report, so the override arm was never
    # actually comparable with the rest of the benchmark -- the
    # "override collapses to 0.00" result rests on a different
    # instrument from distillation's 0.94.
    #
    # The eval here is genuinely teacher-off: greedy_eval builds its
    # own environment and calls only the agent's policy, so the
    # overrider is not consulted no matter what the current override
    # probability is. That makes it meaningful even mid-training,
    # while the teacher is still driving. 0 disables.
    eval_interval: int = 50
    eval_episodes: int = 10
    # Also evaluate the SAMPLED (non-argmax) policy on the same
    # episodes; a deterministic argmax policy can cycle in a gridworld
    # and report 0.00 for a policy that has clearly learned something.
    eval_sampled: bool = True

    # --- Derived at runtime in train(); not set on the CLI ---
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonally initialize a layer's weights and constant-fill its
    bias, the standard PPO initialization.
    """

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Shared-CNN actor-critic, identical to the ppo.py baseline.
    """

    def __init__(self, envs, symbolic=False, recurrent=False):
        """
        Build the encoder, optional recurrent core, and heads.

        `symbolic` selects the integer-grid input path in
        algos/nets.ObsEncoder; False keeps the pixel behaviour
        every existing run used.

        `recurrent` inserts a GRU between the trunk and the heads,
        so an override arm can be measured at the SAME configuration
        as the exploration-bonus arms rather than only on RGB, where
        no method reaches 0.20 and the comparison says nothing.
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
        query must not perturb the state the rollout will use next.
        """

        hidden, _ = self.get_states(x, core_state, episode_start)
        return self.critic(hidden)

    def get_action_and_value(
        self,
        x,
        action=None,
        core_state=None,
        episode_start=None,
        return_probs=False,
    ):
        """
        Sample (or score) an action and return it with its log
        probability, entropy, state value, and updated core state.

        With `return_probs=True` the full action distribution is
        appended to the returned tuple. The advice-budgeting code
        needs it to ask whether the student's policy actually
        favours the teacher's action, which is a better question
        than whether this particular sample happened to match it
        (see advising/policy.probability_mistake). The distribution
        is already computed here, so asking for it costs nothing
        beyond the copy.
        """

        hidden, core_state = self.get_states(
            x, core_state, episode_start
        )
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        out = (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(hidden),
            core_state,
        )
        if return_probs:
            return out + (probs.probs,)
        return out


def make_teacher(
    name, env_id, seed, model='', prompt_id='default',
    reasoning_effort='',
):
    """
    Build the requested teacher behind a common interface.

    All three teachers return the same Advice object, so the override
    mechanism does not change when the teacher does. The oracle is
    free; the llm and vlm teachers call the OpenAI API and need
    OPENAI_API_KEY in the environment.

    Parameters
    ----------
    name: str
        'oracle', 'llm', or 'vlm'.
    env_id: str
        Gymnasium env id (the oracle needs it for grid dimensions;
        the API teachers put it in their prompt header).
    seed: int
        Propagated to the teacher.
    model: str
        Optional model override for the API teachers; empty string
        uses their own default. Ignored by the oracle.
    prompt_id: str
        Prompt variant for the LLM teacher ('default' or 'v2').
        Ignored by the oracle and (for now) the VLM.
    reasoning_effort: str
        Reasoning budget for the gpt-5 / o-series models, passed
        through to the LLM teacher. Empty (the default) sends no
        reasoning field and preserves this file's original
        behaviour exactly. See teachers/minigrid/llm.py.
    """

    if name == 'oracle':
        return MiniGridBFSTeacher(env_id=env_id, seed=seed)
    if name == 'bot':
        # The BabyAI Bot -- free, offline, and (unlike the oracle)
        # able to solve BabyAI-registered tasks including
        # keycorridor_s3r3 / keycorridor_s6r3_babyai. It is stateful
        # (a per-episode plan), so train() below builds ONE instance
        # per parallel env and feeds each the action its env
        # actually executed. See docs/keycorridor_distill_diagnosis.md
        # for why the prefix hand-off with this teacher is the fix
        # for KeyCorridor, where loss-only distillation stalls at 0%.
        from teachers.minigrid.bot_teacher import BabyAIBotTeacher

        return BabyAIBotTeacher(env_id=env_id, seed=seed)
    if name == 'llm':
        from teachers.minigrid.llm import MiniGridLLMTeacher

        kwargs = {'prompt_id': prompt_id}
        if model:
            kwargs['model'] = model
        if reasoning_effort:
            kwargs['reasoning_effort'] = reasoning_effort
        return MiniGridLLMTeacher(env_id=env_id, seed=seed, **kwargs)
    if name == 'vlm':
        from teachers.minigrid.vlm import MiniGridVLMTeacher

        kwargs = {'model': model} if model else {}
        return MiniGridVLMTeacher(env_id=env_id, seed=seed, **kwargs)
    raise ValueError(
        f'unknown teacher {name!r}; choose oracle, bot, llm, or vlm'
    )


def train(args):
    """
    Train a teacher-guided PPO agent and log to TensorBoard.

    The skeleton matches algos/ppo.train; the only new logic is in
    the rollout phase, where some sampled actions are replaced by the
    teacher's before stepping the environments.
    """

    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size

    # Tag JSRL prefix runs so their result folders do not collide with
    # prob-override runs of the same teacher under results/runs/.
    algo_tag = f'ppo_{args.teacher}'
    if args.handoff == 'prefix':
        algo_tag += '_jsrl'
    # Same reasoning as the handoff tag above, for the budgeting
    # settings: without them every arm of a sweep shares a name and
    # arms starting in the same second overwrite each other.
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
        algo_tag += '_' + '_'.join(parts)
    run_name = (
        f'{args.task}__{algo_tag}__'
        f'{args.seed}__{int(time.time())}'
    )

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
    # Dedicated RNG for the per-env override draws.
    rng = np.random.default_rng(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
    )

    # Keep a direct reference to the SyncVectorEnv so we can reach
    # each sub-env's unwrapped MiniGrid state for the teacher. The
    # episode-stats wrapper is layered on top for logging.
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
    ), 'this teacher PPO only supports discrete actions'

    # Build the teacher (needs the real env id for grid dimensions)
    # and the override schedule.
    env_id = TASKS[args.task]
    # The BabyAI Bot is stateful (a per-episode plan tracking the
    # student's real trajectory), so it needs ONE instance per
    # parallel env, plus per-env bookkeeping; the other teachers are
    # stateless and a single shared instance is enough. This is the
    # same split algos/ppo_distill.py already makes.
    is_bot = args.teacher == 'bot'
    if is_bot:
        if not hasattr(sync_envs.envs[0].unwrapped, 'instrs'):
            raise ValueError(
                f'task {args.task!r} is not a BabyAI-registered '
                'level (no .instrs); the bot teacher cannot advise '
                'it. Use a BabyAI-registered task id (e.g. '
                'keycorridor_s3r3, keycorridor_s6r3_babyai, gotoseq).'
            )
        teachers = [
            make_teacher('bot', env_id, args.seed + i)
            for i in range(args.num_envs)
        ]
        # Whether each env just started a fresh episode (rebuild the
        # bot's plan) and the action the student last executed there
        # (fed to the bot so its plan tracks reality).
        fresh_episode = [True] * args.num_envs
        last_actions = [None] * args.num_envs
        # The bot arm keeps one teacher PER ENV in `teachers`, so the
        # single-teacher name has to exist and be None: code after
        # the rollout asks whether the teacher was wrapped for free
        # peeking, and that question has to be answerable on this
        # branch too rather than raising UnboundLocalError.
        teacher = None
    else:
        if args.teacher == 'oracle' and 'doorkey' not in args.task:
            raise ValueError(
                'the BFS oracle only supports doorkey_* tasks; use '
                '--teacher bot for BabyAI-registered tasks '
                '(keycorridor_s3r3, gotoseq, ...).'
            )
        teacher = make_teacher(
            args.teacher,
            env_id,
            args.seed,
            model=args.teacher_model,
            prompt_id=args.teacher_prompt,
        )
    # The VLM teacher reasons over the rendered map, so it needs an
    # image in its context; the oracle and LLM teachers do not.
    needs_image = args.teacher == 'vlm'
    # Both hand-off strategies share the Advice/apply contract but
    # differ in WHICH steps the teacher drives (see integration/
    # override.py). 'prefix' is the Jump-Start fix for the transfer
    # collapse seen with 'prob'.
    prefix_mode = args.handoff == 'prefix'
    if prefix_mode:
        overrider = PrefixOverrider(
            start_len=args.handoff_start_len,
            end_len=args.handoff_end_len,
            fraction=args.override_fraction,
            total_timesteps=args.total_timesteps,
        )
    else:
        overrider = ActionOverrider(
            start_prob=args.override_start,
            end_prob=args.override_end,
            fraction=args.override_fraction,
            total_timesteps=args.total_timesteps,
        )

    # The advice-budgeting strategy sits inside the hand-off mask as
    # a second gate. Importance comes from the policy rather than a
    # Q-table (see advising/policy.py); it is passed as a ready-made
    # function so the advising package itself stays identical to the
    # copy in the sibling tabular repo.
    # All of these except 'entropy_nats' and 'logit_gap' already
    # live in [0, 1] and mean the same thing whatever the action
    # count, so a threshold tuned on DoorKey carries over to
    # KeyCorridor. The two raw ones do not and are kept for
    # comparison against the paper's formulation.
    policy_importance = {
        'none': None,
        'entropy': normalized_entropy_importance,
        'entropy_nats': policy_entropy_importance,
        'max_prob': max_prob_importance,
        'top2_gap': top2_gap_importance,
        'logit_gap': logit_gap_importance,
    }
    if args.importance_source not in policy_importance:
        # A bare KeyError here would name the missing key and nothing
        # else, which is unhelpful for a name that is legitimate
        # elsewhere in the project -- 'teacher_q' and 'student_q' are
        # valid in the tabular controllers but have no meaning for a
        # PPO student with no Q-table.
        raise ValueError(
            f'--importance-source {args.importance_source!r} is not '
            f'available here; expected one of '
            f'{sorted(policy_importance)}. A PPO student has no '
            f'Q-table, so the tabular sources (student_q, '
            f'teacher_q) do not apply.'
        )
    importance_fn = policy_importance[args.importance_source]

    # The predictor-based strategies key states through a hashable
    # state key. The stateful bot teacher is handed a live env
    # object rather than a symbolic state, which has no meaningful
    # key, so those strategies are refused for it up front instead
    # of silently never predicting anything.
    if is_bot and args.advisor in ('predictive', 'surrogate'):
        raise ValueError(
            f'--advisor {args.advisor} needs a hashable state key, '
            'but the bot teacher is driven from a live env object. '
            'Use --teacher oracle/llm/vlm, or an advisor without a '
            'predictor (early, importance, mistake).'
        )

    # Wrap the teacher so mistake correcting and predictive advising
    # can check its action before paying for it. Without this they
    # cannot save queries against an API teacher at all -- see the
    # `peek` field's comment and advising/peekable.py.
    if args.peek not in ('none', 'blind', 'exact', 'surrogate'):
        # Without this, an unrecognized value fell through to the
        # full surrogate wrapper -- a typo would silently enable
        # learned guessing and the run would look like it had been
        # configured deliberately.
        raise ValueError(
            f'--peek {args.peek!r} is not valid; expected none, '
            f'blind, exact, or surrogate.'
        )
    if args.peek != 'none':
        if is_bot:
            raise ValueError(
                '--peek needs a hashable state key, but the bot '
                'teacher is driven from a live env object. Use '
                '--teacher oracle/llm/vlm, or --peek none.'
            )
        teacher = PeekableTeacher(
            teacher,
            num_actions=envs.single_action_space.n,
            predictor_kind=args.predictor,
            trust=args.peek_trust,
            # 'blind' and 'exact' both switch off the learned model;
            # only 'blind' also switches off recall, leaving a
            # wrapper that can never answer a peek.
            exact_only=args.peek in ('exact', 'blind'),
            recall=args.peek != 'blind',
            # Every wrapper mode faces the same teacher, so any
            # difference between the arms is the wrapper's doing.
            use_inner_peek=False,
        )

    # A threshold of 0 keeps the paper's exact-action comparison;
    # anything above it switches to the probability test, which is
    # the definition suited to a student that samples.
    mistake_fn = (
        probability_mistake(args.mistake_threshold)
        if args.mistake_threshold > 0.0
        else None
    )

    advisor = make_advisor(
        args.advisor,
        num_actions=envs.single_action_space.n,
        advice_budget=args.advice_budget,
        query_budget=args.query_budget,
        threshold=args.importance_threshold,
        importance_fn=importance_fn,
        normalize_importance=not args.raw_importance,
        target_rate=(
            args.advice_rate if args.advice_rate > 0.0 else None
        ),
        # With a budget and no explicit rate, the advisor paces
        # itself to spend exactly the budget across exactly this
        # many environment steps -- so every arm of a comparison
        # delivers the same amount of advice, evenly spread, with
        # no rate to reconcile against the cap.
        horizon=args.total_timesteps,
        mistake_fn=mistake_fn,
        predictor_kind=args.predictor,
        trust=args.surrogate_trust,
    )

    # The observation mode decides how the trunk reads the
    # observation, so derive it once here.
    symbolic = is_symbolic_mode(args.obs_mode)
    agent = Agent(
        envs, symbolic=symbolic, recurrent=args.recurrent
    ).to(device)
    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5
    )

    # A recurrent update replays each env's contiguous sequence, so
    # minibatches are groups of ENVS; that only divides evenly when
    # num_envs does.
    if args.recurrent:
        assert args.num_envs % args.num_minibatches == 0, (
            f'recurrent minibatching needs num_envs '
            f'({args.num_envs}) divisible by num_minibatches '
            f'({args.num_minibatches})'
        )
        envs_per_batch = args.num_envs // args.num_minibatches

    obs_shape = envs.single_observation_space.shape
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
    # episode_starts is `dones` shifted one step, which is what the
    # recurrent core consumes; see algos/nets.RecurrentCore for why
    # the two are not the same array.
    episode_starts = torch.zeros(
        (args.num_steps, args.num_envs)
    ).to(device)

    global_step = 0
    start_time = tracker.start_time
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    # The first observation of training genuinely follows a reset.
    episode_start = torch.ones(args.num_envs).to(device)
    core_state = agent.initial_core_state(args.num_envs, device)
    # Per-env step counter within the current episode, used only by
    # the prefix hand-off (override while ep_step < H). Reset to 0
    # whenever an env's episode finishes.
    ep_step = np.zeros(args.num_envs, dtype=int)

    recent_returns = deque(maxlen=100)
    recent_successes = deque(maxlen=100)
    # Most recent teacher-off eval, carried between iterations so the
    # run summary always reports the latest measurement.
    last_eval = None
    # Running totals for teacher usage, reported to TensorBoard.
    total_overrides = 0
    total_queries = 0
    total_cost = Cost()
    # Sliding window of recent episode lengths, used only to
    # estimate how much of a typical episode the guide prefix still
    # covers. A window rather than a lifetime mean because episodes
    # shorten sharply as the agent learns.
    recent_ep_lengths = deque(maxlen=100)

    for iteration in range(1, args.num_iterations + 1):

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]['lr'] = frac * args.learning_rate

        # The update replays each minibatch's stored sequence from
        # exactly the state this rollout started in, so snapshot it
        # before the rollout advances it.
        initial_core_state = (
            core_state.detach().clone()
            if core_state is not None
            else None
        )

        # --- Rollout collection (with teacher override) ---
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            episode_starts[step] = episode_start

            # Keep the state this step STARTED from. The executed
            # action's log-prob is recomputed below on the same
            # observation, and feeding it the already-advanced state
            # would step the GRU twice for one environment step.
            prev_core = core_state
            prev_start = episode_start

            # Sample the agent's own action first. Entropy is kept
            # (it used to be discarded) because it is the policy's
            # free stand-in for the paper's Q-spread: high entropy
            # means the student has no idea what to do here, which
            # is where a paid query buys the most.
            with torch.no_grad():
                action, _, entropy, value, core_state, act_probs = (
                    agent.get_action_and_value(
                        next_obs,
                        core_state=prev_core,
                        episode_start=prev_start,
                        return_probs=True,
                    )
                )
                values[step] = value.flatten()
                entropy_np = entropy.cpu().numpy()
                # Per-env action distributions, used by the
                # probability-based mistake test.
                act_probs_np = act_probs.cpu().numpy()

            # dones[step] (just stored) says whether the NEXT
            # observation begins a fresh episode.
            episode_start = dones[step].clone()

            # Decide which envs to override this step, then query the
            # teacher only for those. The teacher reads the symbolic
            # state pulled from each sub-env; the agent never sees it.
            # Prefix mode overrides the first H steps of each episode;
            # prob mode flips a per-env coin. Both return a bool mask.
            if prefix_mode:
                flags = overrider.select(ep_step, global_step)
            else:
                flags = overrider.select(
                    args.num_envs, global_step, rng
                )
            advices = [None] * args.num_envs
            sampled_np = action.cpu().numpy()

            # Every env advances one step this iteration whether or
            # not the advisor was consulted on it. Reporting that is
            # what makes the advisor's rates a cost per environment
            # step rather than a fraction of whatever the hand-off
            # schedule happened to show it.
            advisor.note_env_steps(args.num_envs)

            # Tell the advisor how much opportunity is left. Only
            # this loop knows the hand-off schedule, and under a
            # prefix hand-off the answer falls to zero long before
            # training ends -- an advisor left to infer it from past
            # visibility paces as if the runway were endless and is
            # still holding most of its budget when the teacher
            # stops being consulted.
            mean_ep_len = (
                float(np.mean(recent_ep_lengths))
                if recent_ep_lengths
                else None
            )
            # `global_step` already advances by num_envs each step,
            # so the integration below is in environment-step units
            # and must NOT be scaled by num_envs again.
            advisor.set_expected_remaining_visible(
                overrider.expected_remaining_visible(
                    global_step, args.num_envs, mean_ep_len
                )
            )
            # Which envs actually spent a query this step. The bot is
            # stateful and replans off the student's last action, so
            # any step it was NOT consulted on leaves its plan a step
            # behind reality; this records the gaps so they can be
            # repaired below.
            queried_now = np.zeros(args.num_envs, dtype=bool)

            # Budgets are global across envs and are decremented as
            # this loop runs, so when one delivery remains the env
            # visited first wins it. A fixed order would hand that
            # advantage to env 0 on every contested step for the
            # whole run, and envs differ by seed, so their state
            # distributions would be taught unequally. Shuffling
            # costs nothing and removes the bias.
            env_order = rng.permutation(args.num_envs)
            for i in env_order:
                i = int(i)
                if not flags[i]:
                    continue

                # The advisor is the second gate: the overrider said
                # this step MAY be driven by the teacher, and the
                # advisor decides whether it is worth paying for.
                advisor.note_step(int(ep_step[i]))
                ctx = {
                    'entropy': float(entropy_np[i]),
                    'action_probs': act_probs_np[i],
                }
                student_action_i = int(sampled_np[i])
                # Remembered so the bot branch below can tell
                # whether the advisor actually spent a query.
                queries_before = advisor.num_asked

                u_i = sync_envs.envs[i].unwrapped
                if is_bot:
                    # The stateful bot reads the live env and needs
                    # to know whether this is a fresh episode and
                    # what the student last did. In prefix mode the
                    # flagged steps are the contiguous start of each
                    # episode, so the bot sees an unbroken action
                    # chain and stays synchronized.
                    #
                    # The bot is stateful: it holds a plan and
                    # advances it using the action the student last
                    # executed. Skipping a query therefore desyncs
                    # it, because the student keeps acting while the
                    # bot's plan does not advance. `fresh_episode`
                    # doubles as the repair signal -- it makes the
                    # bot discard its plan and rebuild from the live
                    # env, which is exactly the right recovery from a
                    # gap of any length.
                    ctx.update(
                        {
                            'new_episode': fresh_episode[i],
                            'last_action': last_actions[i],
                        }
                    )
                    advice, advice_cost = advisor.advise(
                        teachers[i], u_i, student_action_i, ctx
                    )
                    if advisor.num_asked > queries_before:
                        # A real query happened, so the bot is now
                        # rebuilt or advanced and back in sync.
                        fresh_episode[i] = False
                        queried_now[i] = True
                else:
                    state_i = extract_doorkey_state(u_i)
                    # Build the teacher's context: the mission
                    # always, and the full-map render only for the
                    # VLM (it reasons over pixels, not the symbolic
                    # state).
                    ctx['mission'] = getattr(u_i, 'mission', '')
                    if needs_image:
                        ctx['image'] = u_i.get_frame(
                            highlight=False, tile_size=32
                        )
                    advice, advice_cost = advisor.advise(
                        teacher, state_i, student_action_i, ctx
                    )
                    # Let the advisor watch the step. Observation is
                    # free -- the paper's teacher sees everything the
                    # student does and budgets only what it says
                    # about it. The action passed is the one the
                    # policy SAMPLED, before any override, so a
                    # student-action predictor learns the student and
                    # not the teacher.
                    #
                    # The advice is deliberately NOT passed on. The
                    # surrogate strategy already trains on every
                    # answer inside its own deliver gate, so handing
                    # it the same answer again would train its model
                    # twice on one observation and report an
                    # accuracy and coverage that no single pass
                    # earned.
                    advisor.observe(state_i, student_action_i)
                advices[i] = advice
                # The advisor returns the cost even when it withholds
                # the answer, because a discarded LLM call is still
                # billed. `total_queries` counts consultations, which
                # is now what the advisor tracks.
                total_cost = total_cost + advice_cost
                total_queries = advisor.num_asked

            # Build the executed actions (teacher action where
            # overridden, agent action otherwise).
            executed, n_over = overrider.apply(
                action.cpu().numpy(), advices, flags
            )
            total_overrides += n_over
            executed_t = torch.tensor(executed, device=device).long()

            # Recompute the log-prob of the ACTUALLY executed action
            # under the current policy, so PPO's ratio is computed
            # against what was really done (overridden or not).
            with torch.no_grad():
                # Same observation and same starting state as the
                # sample above -- only the scored action differs, so
                # the returned core state is discarded rather than
                # carried forward.
                _, exec_logprob, _, _, _ = (
                    agent.get_action_and_value(
                        next_obs,
                        executed_t,
                        core_state=prev_core,
                        episode_start=prev_start,
                    )
                )
            actions[step] = executed_t
            logprobs[step] = exec_logprob

            next_obs, reward, terminations, truncations, infos = (
                envs.step(executed)
            )
            done_mask = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done_mask).to(device)

            # Advance each env's within-episode step counter, resetting
            # the envs that just finished so the next episode's prefix
            # starts at 0. Only the prefix hand-off reads ep_step.
            ep_step = np.where(done_mask, 0, ep_step + 1)

            # Keep the stateful bot synchronized: remember what the
            # student actually executed (its plan replans off this),
            # and flag envs whose episode just ended so the next
            # query rebuilds a fresh plan for the new layout.
            if is_bot:
                for i in range(args.num_envs):
                    last_actions[i] = int(executed[i])
                    # Any step this env was not consulted on leaves
                    # the bot's plan one action behind the student.
                    # Flagging a rebuild is the only sound repair:
                    # `last_action` carries a single action, so it
                    # cannot describe a gap of several steps, and
                    # feeding it one action after the student took
                    # three would advance the plan to the wrong
                    # place while looking like it worked.
                    if not queried_now[i]:
                        fresh_episode[i] = True
                    if done_mask[i]:
                        fresh_episode[i] = True
                        last_actions[i] = None

            if 'episode' in infos:
                mask = infos['_episode']
                for i in range(args.num_envs):
                    if not mask[i]:
                        continue
                    ep_return = float(infos['episode']['r'][i])
                    ep_length = float(infos['episode']['l'][i])
                    # Kept so the prefix hand-off can estimate what
                    # fraction of a typical episode it still covers,
                    # which is what tells the advisor how much
                    # opportunity remains.
                    recent_ep_lengths.append(ep_length)
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

        # --- Advantage estimation (GAE), identical to ppo.py ---
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                core_state=core_state,
                episode_start=episode_start,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * nextnonterminal
                    * lastgaelam
                )
            returns = advantages + values

        # Refit the advisor's action predictors between rollouts.
        # The paper retrains its classifier between episodes rather
        # than during them; a PPO iteration is the vectorized
        # equivalent of that boundary -- envs run asynchronously, so
        # there is no shared per-episode moment to refit at.
        # Strategies without a predictor ignore this.
        advisor.note_episode_end(iteration)

        # The peek wrapper keeps its own model of the teacher, so it
        # refits on the same boundary.
        if isinstance(teacher, PeekableTeacher):
            teacher.fit()

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_episode_starts = episode_starts.reshape(-1)

        # flat_grid[t, e] locates obs[t, e] in the flat arrays, which
        # is how a recurrent minibatch selects every timestep of a
        # chosen group of envs in the right (T-major) order.
        flat_grid = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs
        )

        # --- PPO update, identical to ppo.py ---
        b_inds = np.arange(args.batch_size)
        env_inds = np.arange(args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            # A recurrent core needs temporally contiguous data, so
            # minibatches are whole ENVS there. The feed-forward path
            # keeps the original flat timestep shuffle, so existing
            # runs are unaffected.
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

                _, newlogprob, entropy, newvalue, _ = (
                    agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                        core_state=mb_core,
                        episode_start=mb_starts,
                    )
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                # Clamp the log-ratio before exponentiating so an
                # extreme importance ratio cannot blow the update up
                # into NaNs. This matters most here: an overridden
                # (off-policy) action can get a huge ratio once the
                # policy sharpens, which diverged a doorkey_8x8 run.
                logratio = torch.clamp(logratio, -10.0, 10.0)
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = (
                    0.5
                    * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                )

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + v_loss * args.vf_coef
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                optimizer.step()

        # --- Teacher-off evaluation ---
        # The agent alone, on fixed held-out episodes, with the
        # teacher not consulted at all. This is the number that is
        # comparable with every other algorithm in the benchmark.
        run_eval = args.eval_interval > 0 and (
            iteration % args.eval_interval == 0
            or iteration == args.num_iterations
        )
        if run_eval:

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

            def make_action_fn(greedy):
                """
                Build an action closure from the agent's own policy.

                For a recurrent policy the closure carries a hidden
                state and exposes `.reset`, which greedy_eval calls
                after each env.reset(). greedy_eval runs episodes
                strictly one at a time, so episode_start can always
                be zero here -- reset() has already zeroed the state
                at exactly the right moment.
                """

                state = {'core': agent.initial_core_state(1, device)}

                def reset():
                    state['core'] = agent.initial_core_state(
                        1, device
                    )

                def action_fn(obs_single):
                    with torch.no_grad():
                        tensor = (
                            torch.Tensor(obs_single)
                            .unsqueeze(0)
                            .to(device)
                        )
                        zero_start = torch.zeros(1, device=device)
                        hidden, state['core'] = agent.get_states(
                            tensor, state['core'], zero_start
                        )
                        logits = agent.actor(hidden)
                        if greedy:
                            return int(logits.argmax(dim=-1).item())
                        return int(
                            Categorical(logits=logits).sample().item()
                        )

                if agent.recurrent:
                    action_fn.reset = reset
                return action_fn

            greedy_action = make_action_fn(greedy=True)
            sampled_action = make_action_fn(greedy=False)

            last_eval = greedy_eval(
                greedy_action,
                make_env=make_eval_env,
                num_episodes=args.eval_episodes,
                seed_base=args.seed + 50_000,
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
            if args.eval_sampled:
                sampled = greedy_eval(
                    sampled_action,
                    make_env=make_eval_env,
                    num_episodes=args.eval_episodes,
                    seed_base=args.seed + 50_000,
                )
                last_eval['sampled_success_rate'] = sampled[
                    'success_rate'
                ]
                writer.add_scalar(
                    'charts/eval_sampled_success_rate',
                    sampled['success_rate'],
                    global_step,
                )

        # --- Diagnostics ---
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
        # Teacher-usage signals: the live override probability and the
        # cumulative number of overrides, queries, and teacher cost.
        # Log the live guidance intensity: prefix length H for JSRL,
        # override probability for the coin-flip method.
        if prefix_mode:
            writer.add_scalar(
                'teacher/guide_prefix_len',
                overrider.current_len(global_step),
                global_step,
            )
        else:
            writer.add_scalar(
                'teacher/override_prob',
                overrider.current_prob(global_step),
                global_step,
            )
        writer.add_scalar(
            'teacher/total_overrides', total_overrides, global_step
        )
        writer.add_scalar(
            'teacher/total_queries', total_queries, global_step
        )
        writer.add_scalar(
            'teacher/cost_dollars', total_cost.dollars, global_step
        )
        # Advice actually delivered, which under a budgeting strategy
        # is no longer the same number as queries bought. The gap
        # between these two curves is what the budgeting experiments
        # are about.
        writer.add_scalar(
            'teacher/total_advice', advisor.num_delivered, global_step
        )
        writer.add_scalar(
            'teacher/waste_rate',
            advisor.stats()['waste_rate'],
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
        # Show the live guidance intensity in the summary and console:
        # prefix length H for JSRL, override probability for prob mode.
        if prefix_mode:
            intensity = f'H {overrider.current_len(global_step):4.1f}'
        else:
            intensity = (
                f'ovr {overrider.current_prob(global_step):4.2f}'
            )
        tracker.write_summary(
            status='running',
            global_step=global_step,
            extra={
                'learning_rate': optimizer.param_groups[0]['lr'],
                'value_loss': v_loss.item(),
                'policy_loss': pg_loss.item(),
                'entropy': entropy_loss.item(),
                'approx_kl': approx_kl.item(),
                'clipfrac': float(np.mean(clipfracs)),
                'teacher_total_overrides': int(total_overrides),
                'teacher_total_queries': int(total_queries),
                'teacher_cost_dollars': total_cost.dollars,
                'teacher_wall_time_s': total_cost.wall_time_s,
                'teacher_tokens_in': int(total_cost.tokens_in),
                'teacher_tokens_out': int(total_cost.tokens_out),
                'teacher_compute_units': int(total_cost.compute_units),
                # The teacher-off eval, so this arm's summary
                # carries the same fields every other algo writes.
                **(
                    {}
                    if last_eval is None
                    else {
                        'eval_success_rate':
                            last_eval['success_rate'],
                        'eval_mean_return':
                            last_eval['mean_return'],
                        **(
                            {
                                'eval_sampled_success_rate':
                                    last_eval['sampled_success_rate']
                            }
                            if 'sampled_success_rate' in last_eval
                            else {}
                        ),
                    }
                ),
                'guidance_intensity': intensity,
            },
        )
        if recent_returns:
            avg_succ = sum(recent_successes) / len(recent_successes)
            avg_ret = sum(recent_returns) / len(recent_returns)
            progress = f'success {avg_succ:4.2f} return {avg_ret:6.3f}'
        else:
            progress = 'success  --  return    --'
        print(
            f'iter {iteration}/{args.num_iterations} '
            f'step {global_step} SPS {sps} | {progress} '
            f'| {intensity}'
        )

    # RunTracker.summary() REPLACES its metrics dict whenever `extra`
    # is passed, so the final close() must repeat the teacher-off eval
    # fields -- otherwise this last write silently erases the
    # eval_success_rate that write_summary recorded during training,
    # and the run ends with no comparable score at all.
    final_extra = {
        'teacher_total_overrides': int(total_overrides),
        'teacher_total_queries': int(total_queries),
        'teacher_cost_dollars': total_cost.dollars,
        'teacher_wall_time_s': total_cost.wall_time_s,
        'teacher_tokens_in': int(total_cost.tokens_in),
        'teacher_tokens_out': int(total_cost.tokens_out),
        'teacher_compute_units': int(total_cost.compute_units),
        # Full budget accounting: how much advice was delivered, how
        # much was paid for and discarded, and where the budget went.
        'advising': advisor.stats(),
    }
    # Attribution of the free lookups, when the teacher was wrapped:
    # how many peeks were answered from real recorded answers versus
    # guessed by the model.
    if isinstance(teacher, PeekableTeacher):
        final_extra['peekable'] = teacher.stats()
    if last_eval is not None:
        final_extra['eval_success_rate'] = last_eval['success_rate']
        final_extra['eval_mean_return'] = last_eval['mean_return']
        if 'sampled_success_rate' in last_eval:
            final_extra['eval_sampled_success_rate'] = last_eval[
                'sampled_success_rate'
            ]
    tracker.close(
        status='completed',
        global_step=global_step,
        extra=final_extra,
    )
    # Save the trained policy next to the run summary, matching
    # algos/ppo.py, so this arm can be re-evaluated later without
    # being retrained.
    torch.save(agent.state_dict(), os.path.join(run_dir, 'agent.pt'))
    envs.close()
    writer.close()


if __name__ == '__main__':
    train(tyro.cli(Args))
