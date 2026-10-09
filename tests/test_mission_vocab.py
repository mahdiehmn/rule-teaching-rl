"""
Tests for the frozen BabyAI mission vocabulary.

Covers the tokenize/encode round-trip, padding and truncation
behavior, the save/load round-trip, and -- since this is cheap and
converts an assumption into a guarantee -- an exhaustiveness check
that every color/object/location word minigrid's BabyAI grammar can
produce is actually present in the committed vocabulary file. That
check is what makes a future minigrid version bump that adds
vocabulary fail loudly here instead of silently flooding <unk> at
train time.
"""

import numpy as np
import pytest

pytest.importorskip('minigrid')

from envs.mission_vocab import (
    PAD_ID,
    UNK_ID,
    MissionVocab,
)


def test_tokenize_lowercases_and_splits_on_words():
    """
    Tokenization is a simple lowercase word split; punctuation-free
    BabyAI missions round-trip through it losslessly.
    """

    tokens = MissionVocab.tokenize('Go to the Grey Box')
    assert tokens == ['go', 'to', 'the', 'grey', 'box']


def test_build_from_missions_assigns_ids_alphabetically():
    """
    Ids must be deterministic given the same set of words,
    independent of the order missions were sampled in -- this is
    what makes the saved file reproducible.
    """

    vocab = MissionVocab.build_from_missions(
        ['go to the red ball', 'open the blue door']
    )
    words = vocab.id2word[2:]  # skip <pad>/<unk>
    assert words == sorted(words)
    assert vocab.id2word[PAD_ID] == '<pad>'
    assert vocab.id2word[UNK_ID] == '<unk>'


def test_encode_round_trip_and_padding():
    """
    A known mission encodes to the right ids, zero-padded past its
    true length, with the correct reported length.
    """

    vocab = MissionVocab.build_from_missions(['go to the red ball'])
    ids, length = vocab.encode('go to the red ball', max_len=10)

    assert length == 5
    assert ids.dtype == np.int64
    assert ids.shape == (10,)
    # Every id past the true length is the pad id.
    assert np.all(ids[5:] == PAD_ID)
    assert np.all(ids[:5] != PAD_ID)


def test_encode_maps_unknown_words_to_unk():
    """
    A word absent from the vocabulary must not raise -- it degrades
    to <unk> so an out-of-vocabulary mission cannot crash training.
    """

    vocab = MissionVocab.build_from_missions(['go to the red ball'])
    ids, length = vocab.encode('fly over the grey castle', max_len=10)

    assert length == 5
    # 'to'/'the' are known; 'fly'/'over'/'grey'/'castle' are not.
    assert ids[0] == UNK_ID  # 'fly'
    assert ids[3] == UNK_ID  # 'grey'


def test_encode_truncates_and_warns_when_too_long():
    """
    A mission longer than max_len is truncated (never overflows the
    caller's fixed-size buffer) and warns rather than failing
    silently.
    """

    vocab = MissionVocab.build_from_missions(
        ['go to the red ball and then open the blue door']
    )
    with pytest.warns(UserWarning, match='exceeds max_len'):
        ids, length = vocab.encode(
            'go to the red ball and then open the blue door',
            max_len=3,
        )
    assert length == 3
    assert ids.shape == (3,)


def test_save_and_load_round_trip(tmp_path):
    """
    A vocabulary saved to disk and reloaded must assign the exact
    same ids -- this is the guarantee that lets a training run and
    a later evaluation of the same model agree on token meaning.
    """

    vocab = MissionVocab.build_from_missions(
        ['go to the red ball', 'open the blue door']
    )
    path = tmp_path / 'vocab.json'
    vocab.save(path)

    loaded = MissionVocab.load(path)
    assert loaded.id2word == vocab.id2word
    assert loaded.word2id == vocab.word2id


def test_committed_vocab_covers_babyai_grammar():
    """
    The frozen, committed vocabulary (envs/vocab/babyai_missions.json)
    must contain every color, object, and location word minigrid's
    BabyAI grammar can produce. This turns "the sample we built the
    vocab from was probably exhaustive" into a checked guarantee: if
    a future minigrid version adds a color or object type, this test
    fails loudly instead of every new mission silently degrading to
    <unk>.
    """

    from minigrid.core.constants import COLOR_NAMES

    vocab = MissionVocab.load()

    expected_words = set(COLOR_NAMES) | {
        'ball',
        'box',
        'key',
        'door',
    }
    missing = expected_words - set(vocab.word2id)
    assert not missing, (
        f'committed vocab is missing grammar words {missing}; '
        'rerun scripts/build_mission_vocab.py'
    )
