"""The project-asset featurizer (preprocess.py, loaded by the pipeline by path)
must produce byte-identical features to the canonical library featurizer
(src/subjective_randomness/features.py). The two were duplicated copies; this
guard pins them to one behavior so they can never silently drift — the human
experiment featurizes via preprocess.py and the model-recovery loop generates
synthetic responses via features.py, so a divergence would make the two loops
incomparable.
"""

from __future__ import annotations

from itertools import product

import pytest

from src.subjective_randomness.features import featurize_stimulus as feat_library
from tests.paths import REPO_ROOT, load_script_module

PREPROCESS = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _load_preprocess_by_path():
    """Load preprocess.py exactly as the pipeline does (by file path)."""
    return load_script_module(PREPROCESS, "_sr_preprocess_parity")


def _all_sequences(max_len: int):
    seqs = []
    for length in range(1, max_len + 1):
        seqs.extend("".join(bits) for bits in product("HT", repeat=length))
    return seqs


def test_preprocess_featurizer_matches_library_over_battery():
    feat_project = _load_preprocess_by_path().featurize_stimulus
    seqs = _all_sequences(6)  # every H/T sequence up to length 6
    # Pair each sequence with itself and with its reverse: covers symmetric and
    # asymmetric a/b pairs, all lengths, all-H/all-T, alternating, and runs.
    pairs = [(s, s) for s in seqs] + [(s, s[::-1]) for s in seqs]
    for seq_a, seq_b in pairs:
        assert feat_project(seq_a, seq_b) == feat_library(seq_a, seq_b), (seq_a, seq_b)


def test_preprocess_featurizer_rejects_empty_sequences_like_the_library():
    # The degenerate case has to stay in parity too: both raise rather than
    # emitting a zero-filled row (see the featurizer's clean_sequence).
    feat_project = _load_preprocess_by_path().featurize_stimulus
    for pair in (("", "HT"), ("HT", ""), ("", "")):
        with pytest.raises(ValueError, match="must not be empty"):
            feat_project(*pair)
        with pytest.raises(ValueError, match="must not be empty"):
            feat_library(*pair)
