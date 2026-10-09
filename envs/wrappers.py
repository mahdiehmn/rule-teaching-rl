"""
Observation wrappers shared across the benchmark.

The benchmark deliberately trains on MiniGrid image observations
rather than a hand-extracted grid state. Two wrappers are composed to
get there:

- RGBImgPartialObsWrapper or RGBImgObsWrapper (from minigrid) turns
  the agent's view or full map into an RGB image under obs['image'],
  while keeping obs['mission'].
- ImgObsWrapper (from minigrid) then drops everything except the
  image, so the agent observation is a plain HxWx3 array.

The problem: ImgObsWrapper discards obs['mission'], but the LLM/VLM
teacher needs that natural-language instruction. KeepMissionWrapper
sits between the two and copies the mission into the step/reset info
dict, so the training loop can still hand it to the teacher even
though the agent itself only sees pixels.

For BabyAI (language-conditioned) tasks, pixels alone are not
enough for the STUDENT either -- the mission is often the only
thing that disambiguates the goal (e.g. "go to a grey box" when
several colored boxes are visible), so a vision-only policy cannot
represent the task at all, let alone learn it. MissionTokenWrapper
is the alternative to ImgObsWrapper for those tasks: instead of
dropping the mission, it turns it into a fixed-length token array
the agent's observation actually carries.
"""

import gymnasium as gym
import numpy as np


class KeepMissionWrapper(gym.Wrapper):
    """
    Preserve the MiniGrid mission string in the info dict.

    MiniGrid puts the per-episode instruction in obs['mission'], but
    the image-only wrapper used for the agent throws the rest of the
    observation away. This wrapper captures the mission before that
    happens and republishes it under info['mission'] on both reset
    and step, so teacher code downstream can read it without touching
    the agent's pixel observation.
    """

    def reset(self, **kwargs):
        """
        Reset the env and stash the mission string into info.
        """

        obs, info = self.env.reset(**kwargs)

        # obs is still the full dict here (this wrapper is applied
        # before the image-only wrapper), so the mission is present.
        if isinstance(obs, dict) and 'mission' in obs:
            info['mission'] = obs['mission']

        return obs, info

    def step(self, action):
        """
        Step the env and re-publish the mission string into info.
        """

        obs, reward, terminated, truncated, info = self.env.step(
            action
        )

        # Carry the mission forward every step so the teacher can be
        # queried at any timestep, not just at episode start.
        if isinstance(obs, dict) and 'mission' in obs:
            info['mission'] = obs['mission']

        return obs, reward, terminated, truncated, info


class MissionTokenWrapper(gym.ObservationWrapper):
    """
    Turn obs['mission'] into a fixed-length token array the agent
    can see, instead of dropping it like ImgObsWrapper does.

    Replaces ImgObsWrapper in the wrapper chain for BabyAI tasks:
    where ImgObsWrapper reduces the observation to a bare image
    array, this wrapper keeps a small gym.spaces.Dict with the
    image alongside the tokenized mission, so the student network
    can condition on the language instruction instead of only
    pixels. Requires a MissionVocab built ahead of time (see
    scripts/build_mission_vocab.py) -- the vocabulary must be fixed
    before training starts so token ids never drift between a
    training run and a later evaluation of the same model.

    Unlike HistoricalObsWrapper, this wrapper has no per-episode
    state to reset: it re-derives the token array from
    obs['mission'] on every call, the same way KeepMissionWrapper
    re-publishes info['mission'] every step rather than caching it.
    """

    def __init__(self, env, vocab, max_len):
        """
        Wrap `env` and replace its observation space with a Dict of
        the existing image plus the tokenized mission.

        Parameters
        ----------
        env: gymnasium.Env
            Must still expose obs['image'] and obs['mission'] (i.e.
            wrapped by KeepMissionWrapper and an RGB wrapper, but
            NOT yet by ImgObsWrapper -- this wrapper replaces that
            step).
        vocab: envs.mission_vocab.MissionVocab
            The frozen vocabulary used to tokenize missions.
        max_len: int
            Fixed length every tokenized mission is padded or
            truncated to, so the batched array shape is constant
            across episodes and tasks.
        """

        super().__init__(env)

        self.vocab = vocab
        self.max_len = max_len

        image_space = self.observation_space.spaces['image']
        # mission_ids holds token ids in [0, vocab.size); mission_len
        # is a scalar reporting how many of those ids are real
        # (non-pad) tokens, which the mission encoder's
        # pack_padded_sequence call needs to skip padding entirely.
        self.observation_space = gym.spaces.Dict(
            {
                'image': image_space,
                'mission_ids': gym.spaces.Box(
                    low=0,
                    high=vocab.size - 1,
                    shape=(max_len,),
                    dtype=np.int64,
                ),
                'mission_len': gym.spaces.Box(
                    low=0, high=max_len, shape=(), dtype=np.int64
                ),
            }
        )

    def observation(self, obs):
        """
        Replace the observation dict with {image, mission_ids,
        mission_len}, dropping the raw mission string and anything
        else the upstream wrappers left in obs.
        """

        ids, length = self.vocab.encode(obs['mission'], self.max_len)
        return {
            'image': obs['image'],
            'mission_ids': ids,
            'mission_len': np.int64(length),
        }
