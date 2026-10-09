"""
Reward shaping from teacher advice.

Turns an Advice into an extra reward term added to the environment
reward during training. The recommended form is potential-based
shaping (Ng et al., 1999): shaping the reward as a difference of a
potential function leaves the set of optimal policies unchanged, so
the teacher can speed up learning without changing what counts as
success. A simpler "bonus when the agent follows the advised action"
form is also possible but can bias the optimal policy -- a tradeoff
to document when this is implemented.

Status: STUB.
"""
