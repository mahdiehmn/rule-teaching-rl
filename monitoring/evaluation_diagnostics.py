"""Optional measurements that leave training random streams unchanged."""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


def diagnostic_milestones(spec, batch_size, num_iterations):
    """
    Map requested frames to the first completed rollout at or above them.
    """

    milestones = {}
    for token in spec.split(','):
        if not token.strip():
            continue
        frames = int(token)
        iteration = (frames + batch_size - 1) // batch_size
        if frames < 0 or iteration > num_iterations:
            raise ValueError('Diagnostic frames must fit the training horizon')
        # Keep requested counts even when several share a rollout.
        milestones.setdefault(iteration, set()).add(frames)
    return {key: sorted(values) for key, values in milestones.items()}


def policy_sha256(agent):
    """
    Hash parameters and buffers using the initial-policy artifact format.
    """

    digest = hashlib.sha256()
    for name, tensor in sorted(agent.state_dict().items()):
        digest.update(name.encode() + b'\0')
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def isolated_evaluation_rng(seed):
    """
    Restore all training RNG streams even if an evaluation raises.
    """

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    # CPU jobs should not initialize CUDA just to make a measurement.
    devices = (list(range(torch.cuda.device_count()))
               if torch.cuda.is_initialized() else [])
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            # Seed the CPU generator without queuing lazy CUDA seeds.
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                torch.cuda.default_generators[device].manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
