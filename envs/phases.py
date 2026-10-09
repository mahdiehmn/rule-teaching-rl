"""
Coarse task phases, derived from the symbolic state.

These are **grouping metadata** for the R3 within-phase shuffle, and
nothing else. A phase is never an explanation target, is never shown to
the teacher, and is never an input to the policy. The explanation target
stays real LLM text.

The rule is deliberately kept out of `algos/explanation_head.py`: the
shuffler groups by whatever label it is handed, so that a task-specific
definition cannot leak into generic machinery. It is equally deliberately
derived from the **state** rather than from the explanation text -- a
text-derived grouping would inherit the very content the shuffle is meant
to corrupt, and a "within-phase" swap could then preserve meaning by
construction.
"""

from envs.state import extract_generic_state

# The ordered phase names, most-advanced first. The order IS the rule.
S3R3_PHASES = ('reach_target', 'unlock_door', 'seek_key')


def s3r3_phase(unwrapped):
    """
    Return the coarse task stage of a KeyCorridor-S3R3 state.

    **Ordered, first match wins.** The ordering is the whole content of
    the rule, because the conditions are not naturally disjoint:

      1. `reach_target`  -- no door is still locked
      2. `unlock_door`   -- otherwise, the agent is carrying a key
      3. `seek_key`      -- otherwise

    Testing the locked-door condition FIRST is what makes the mapping
    disjoint. An earlier draft defined `seek_key` as "not carrying a
    key", which also matches a state after the door has been unlocked
    and the key dropped; that state satisfied two rules at once and its
    phase depended on undeclared evaluation order.

    Rule 3 is a bare fallback, so the mapping is **total**: every
    symbolic state receives exactly one phase and there is no
    unlabelled case to discover mid-run.
    """

    _, _, _, carrying, objects = extract_generic_state(unwrapped)

    # objects entries are (kind, color, x, y, door_state); door_state is
    # 'open' / 'closed' / 'locked' for doors and '' for everything else.
    any_locked = any(entry[4] == 'locked' for entry in objects)
    if not any_locked:
        return 'reach_target'
    if carrying and carrying.endswith('key'):
        return 'unlock_door'
    return 'seek_key'


def phase_of_states(unwrapped_states):
    """
    Map a sequence of unwrapped envs to phases, preserving order.
    """

    return [s3r3_phase(state) for state in unwrapped_states]
