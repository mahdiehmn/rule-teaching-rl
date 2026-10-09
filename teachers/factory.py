"""
Shared teacher factory.

algos/ppo_teacher.py has its own local make_teacher; that file is
deliberately frozen as the cautionary override baseline, so rather
than modify it, new algorithms (ppo_distill and later) import this
shared factory instead. It supports everything the local one does
plus the BabyAI Bot teacher.

Which teacher advises which task (see docs/stage1_distill.md):

- 'oracle'      -- MiniGridBFSTeacher, DoorKey family only (its
  planner is hard-coded to DoorKey's key/door/goal structure).
- 'bot'         -- BabyAIBotTeacher, BabyAI levels only (needs the
  language mission object `env.instrs`; e.g. gotoseq, gotolocal).
- 'llm' / 'vlm' -- OpenAI-backed, DoorKey-shaped (see teachers/
  minigrid/llm.py, vlm.py); kept frozen as-is.
- 'vlm_general' -- OpenAI-backed VISION teacher, topology-agnostic
  (see teachers/minigrid/vlm_general.py): reasons over the
  rendered map image; works on any MiniGrid/BabyAI task.
- 'llm_general' -- OpenAI-backed TEXT-ONLY teacher, topology-
  agnostic (see teachers/minigrid/llm_general.py): reasons over an
  ASCII map + object list instead of pixels. Same tasks as
  vlm_general; comparing the two on one task isolates perception
  from planning (the DoorKey study's finding).
- 'llm_subgoal' -- OpenAI-backed HYBRID teacher (see teachers/
  minigrid/llm_subgoal.py): the model picks a sub-goal (a target
  cell plus an interaction) and an exact BFS planner walks there.
  Same inputs as llm_general, but the model never chooses a
  primitive movement, so it cannot mis-ground spatially and costs
  one call per sub-goal instead of one per step.

All the OpenAI-backed teachers need internet. Vulcan's compute nodes
are confirmed to have it (2026-07, via Compute Canada support), so
these teachers now also work in cluster training jobs, not only
locally -- see scripts/submit_cc.sh. On a different Alliance cluster,
re-verify before assuming the same.
"""

from teachers.minigrid.bfs_solver import MiniGridBFSTeacher
from teachers.minigrid.bot_teacher import BabyAIBotTeacher


def make_teacher(
    name, env_id, seed, model='', prompt_id='default', timeout=None,
    strict=None, reasoning_effort='', rule_bank='', online_rules=None,
):
    """
    Build the requested teacher behind the common Advice contract.

    Parameters
    ----------
    name: str
        'oracle', 'bot', 'llm', 'vlm', 'vlm_general', or
        'llm_general'.
    env_id: str
        Gymnasium env id (the oracle derives grid dimensions from
        it; the API teachers embed it in their prompts; the bot
        records it in its teacher_id).
    seed: int
        Propagated to the teacher.
    model: str
        Optional model override for the API teachers. Ignored by
        the offline teachers.
    prompt_id: str
        Prompt variant for the DoorKey LLM teacher. Ignored by the
        others.
    timeout: float or None
        Per-call client timeout in seconds, passed through to the
        API teachers (each defaults to 60.0 if left None). Some
        Aleph/Vulcan reasoning models genuinely take longer than
        60s to respond once warmed up -- distinct from the separate
        "scaled to zero" cold-start delay, which raises its own
        clear error regardless of this setting. Ignored by the
        offline teachers.
    reasoning_effort: str
        Reasoning budget ('minimal', 'low', 'medium', 'high') for
        the gpt-5 / o-series models, passed through to the per-step
        `llm` teacher. Empty (the default) sends no reasoning field,
        which is required for the gpt-4.x models and leaves the
        reasoning models on their own default. Reasoning tokens bill
        as output tokens, so this is the largest single cost lever
        for a paid teacher -- see teachers/minigrid/llm.py for the
        measured numbers and the competence caveat. Only the `llm`
        teacher accepts it so far; the others ignore it.
    strict: bool or None
        Passed through to the API teachers, each of which already
        implements strict=True (raise on API/parse failure) vs
        strict=False (return an abstaining Advice instead) --
        confirmed identical in llm.py, vlm.py, vlm_general.py,
        llm_general.py and llm_subgoal.py. Left None (each class's
        own default, True) preserves existing callers' behavior
        unchanged; ppo_distill.py passes strict=False explicitly,
        because a single flaky call there (confirmed on Aleph: a
        transient 403, a request timeout, or occasional malformed
        JSON -- see results/slurm/eval_teacher_*.out) must not
        crash a training job that can run for up to 2 days
        (scripts/submit_cc.sh's --time), losing every seed's
        progress over one bad advice query. Ignored by the offline
        teachers (oracle, bot, door_bfs), which have no API call to
        fail this way.
    """

    if name == 'oracle':
        return MiniGridBFSTeacher(env_id=env_id, seed=seed)
    if name == 'bot':
        return BabyAIBotTeacher(env_id=env_id, seed=seed)
    if name == 'llm':
        # Imported lazily so offline runs never touch the OpenAI
        # client (which reads OPENAI_API_KEY at import time).
        from teachers.minigrid.llm import MiniGridLLMTeacher

        kwargs = {'prompt_id': prompt_id}
        if model:
            kwargs['model'] = model
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        if reasoning_effort:
            kwargs['reasoning_effort'] = reasoning_effort
        return MiniGridLLMTeacher(env_id=env_id, seed=seed, **kwargs)
    if name == 'vlm':
        from teachers.minigrid.vlm import MiniGridVLMTeacher

        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        return MiniGridVLMTeacher(env_id=env_id, seed=seed, **kwargs)
    if name == 'vlm_general':
        from teachers.minigrid.vlm_general import (
            MiniGridGeneralVLMTeacher,
        )

        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        return MiniGridGeneralVLMTeacher(
            env_id=env_id, seed=seed, **kwargs
        )
    if name == 'llm_general':
        from teachers.minigrid.llm_general import (
            MiniGridGeneralLLMTeacher,
        )

        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        return MiniGridGeneralLLMTeacher(
            env_id=env_id, seed=seed, **kwargs
        )
    if name == 'llm_subgoal':
        from teachers.minigrid.llm_subgoal import (
            MiniGridSubgoalLLMTeacher,
        )

        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        return MiniGridSubgoalLLMTeacher(
            env_id=env_id, seed=seed, **kwargs
        )
    if name == 'llm_scoped':
        from teachers.minigrid.llm_scoped import ScopedConsultTeacher

        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        if strict is not None:
            kwargs['strict'] = strict
        return ScopedConsultTeacher(env_id=env_id, seed=seed, **kwargs)
    if name == 'llm_rules_online':
        from teachers.minigrid.llm_rules_online import OnlineRuleTeacher

        if online_rules is None:
            raise ValueError('llm_rules_online needs the shared rule state')
        kwargs = {'model': model} if model else {}
        if timeout is not None:
            kwargs['timeout'] = timeout
        return OnlineRuleTeacher(env_id=env_id, seed=seed, state=online_rules,
                                 **kwargs)
    if name == 'door_bfs':
        from teachers.minigrid.door_bfs import DoorOnlyBFSTeacher

        return DoorOnlyBFSTeacher(env_id=env_id, seed=seed)
    if name == 'rule_bank':
        # A frozen, LLM-written rule bank read from the student's view;
        # free during training (teachers/minigrid/rule_bank.py).
        from teachers.minigrid.rule_bank import RuleBankTeacher

        return RuleBankTeacher(rule_bank, seed=seed)
    raise ValueError(
        f'unknown teacher {name!r}; choose oracle, bot, llm, vlm, '
        'vlm_general, llm_general, llm_subgoal, or door_bfs'
    )
