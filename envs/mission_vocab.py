"""
Closed-vocabulary tokenizer for BabyAI mission strings.

BabyAI missions are generated from a small, closed synthetic
grammar (colors, object types, a handful of location and function
words), so a full subword tokenizer is unnecessary: a word-level
vocabulary built once from a large sample of real missions is
exhaustive and stays fixed for the life of a run. This module is
that vocabulary plus the encode/decode logic; the actual sampling
and file-writing lives in scripts/build_mission_vocab.py so this
module has no dependency on envs.registry (which in turn needs to
load a MissionVocab, so importing registry here would create a
cycle).

The saved vocabulary file is a frozen, committed artifact (like a
config file), not something regenerated per run: token-to-id
assignment must stay identical between the run that trained a
model and any later run that evaluates it.
"""

import json
import os
import re
import warnings

import numpy as np

# Reserved ids. Index 0 doubles as the array-fill value for padding,
# so a freshly zeroed numpy array is already a valid all-pad
# encoding.
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
PAD_ID = 0
UNK_ID = 1

# Default location to load/save the frozen vocabulary. Kept next to
# this module so the package is self-contained.
DEFAULT_VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), 'vocab', 'babyai_missions.json'
)

# BabyAI's instruction grammar caps a mission at four leaf clauses
# (a "seq" of two "and"s; see minigrid's LevelGen.rand_instr, which
# restricts a seq's children to non-seq kinds). Working through the
# longest possible surface form of each clause type puts the hard
# ceiling at 72 tokens; 80 leaves headroom without meaningfully
# increasing the cost of the padded array (it is tiny next to a
# 56x56x3 image either way).
MAX_MISSION_LEN = 80

# The regex used to split a mission string into words. Matches
# runs of letters and apostrophes (BabyAI missions contain no
# digits or other punctuation), which is exactly the token shape
# every sampled mission has been built from.
_TOKEN_RE = re.compile(r"[a-zA-Z']+")


class MissionVocab:
    """
    A fixed, closed word-level vocabulary for BabyAI missions.

    Construct via `build_from_missions` (offline, once) or `load`
    (at train/eval time); `MissionVocab.__init__` itself just wraps
    an already-decided token list, so both paths funnel through the
    same object.
    """

    def __init__(self, tokens):
        """
        Wrap an explicit, ordered token list as a vocabulary.

        `tokens[0]` must be PAD_TOKEN and `tokens[1]` must be
        UNK_TOKEN, matching PAD_ID/UNK_ID above; callers normally
        never build this list by hand (use `build_from_missions` or
        `load` instead), so this is intentionally unchecked.
        """

        self.id2word = list(tokens)
        self.word2id = {word: i for i, word in enumerate(self.id2word)}

    @property
    def size(self):
        """
        Number of tokens in the vocabulary, including specials.
        """

        return len(self.id2word)

    @staticmethod
    def tokenize(mission):
        """
        Split a mission string into lowercase word tokens.
        """

        return _TOKEN_RE.findall(mission.lower())

    @classmethod
    def build_from_missions(cls, missions):
        """
        Build a vocabulary from a large sample of mission strings.

        Token ids are assigned in alphabetical order (after the two
        reserved specials), not in the order missions happened to
        be sampled, so the saved file is reproducible regardless of
        which seeds or how many resets were used to build it.
        """

        vocab_words = set()
        for mission in missions:
            vocab_words.update(cls.tokenize(mission))
        return cls([PAD_TOKEN, UNK_TOKEN] + sorted(vocab_words))

    def encode(self, mission, max_len=MAX_MISSION_LEN):
        """
        Turn a mission string into a padded id array plus its
        true (unpadded) length.

        Unknown words map to UNK_ID rather than raising, so a
        vocabulary gap degrades gracefully instead of crashing a
        training run; a mission longer than `max_len` is truncated
        with a warning, since the caller's buffers are sized to
        `max_len` and cannot hold more.

        Returns
        -------
        ids: numpy.ndarray, shape (max_len,), dtype int64
            Token ids, zero-padded (PAD_ID == 0) after the true
            tokens.
        length: int
            Number of real (non-pad) tokens, capped at `max_len`.
        """

        tokens = self.tokenize(mission)
        ids = [self.word2id.get(tok, UNK_ID) for tok in tokens]

        if len(ids) > max_len:
            warnings.warn(
                f'mission exceeds max_len={max_len}, truncating: '
                f'{mission!r}'
            )
            ids = ids[:max_len]

        array = np.zeros(max_len, dtype=np.int64)
        array[: len(ids)] = ids
        return array, len(ids)

    def save(self, path=DEFAULT_VOCAB_PATH):
        """
        Write this vocabulary to a JSON file.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'tokens': self.id2word}, f, indent=2)
            f.write('\n')

    @classmethod
    def load(cls, path=DEFAULT_VOCAB_PATH):
        """
        Load a previously saved vocabulary from a JSON file.
        """

        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['tokens'])
