"""
Action masking from teacher advice.

Uses an Advice's forbidden_actions to remove actions from the agent's
choice set at a given state, typically by setting the corresponding
policy logits to a large negative value before sampling. Unlike
reward shaping, masking is a hard constraint applied at action-
selection time and never appears in the reward, so it cannot bias the
return -- but it does require the teacher to be reliable, since a
wrongly forbidden action is simply unavailable.

Status: STUB.
"""
