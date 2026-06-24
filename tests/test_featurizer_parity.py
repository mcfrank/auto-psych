"""The project-asset featurizer (preprocess.py, loaded by the pipeline by path)
must produce byte-identical features to the canonical library featurizer
(src/subjective_randomness/features.py). The two were duplicated copies; this
guard pins them to one behavior so they can never silently drift — the human
experiment featurizes via preprocess.py and the model-recovery loop generates
synthetic responses via features.py, so a divergence would make the two loops
incomparable.
"""

from __future__ import annotations

import importlib.util
from itertools import product
from pathlib import Path

from src.subjective_randomness.features import featurize_stimulus as feat_library

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESS = (
    REPO_ROOT / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _load_preprocess_by_path():
    """Load preprocess.py exactly as the pipeline does (by file path)."""
    spec = importlib.util.spec_from_file_location("_sr_preprocess_parity", PREPROCESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _all_sequences(max_len: int):
    seqs = [""]
    for length in range(1, max_len + 1):
        seqs.extend("".join(bits) for bits in product("HT", repeat=length))
    return seqs


def test_preprocess_featurizer_matches_library_over_battery():
    feat_project = _load_preprocess_by_path().featurize_stimulus
    seqs = _all_sequences(6)  # every H/T sequence up to length 6, plus the empty string
    # Pair each sequence with itself and with its reverse: covers symmetric and
    # asymmetric a/b pairs, all lengths, all-H/all-T, alternating, and runs.
    pairs = [(s, s) for s in seqs] + [(s, s[::-1]) for s in seqs]
    for seq_a, seq_b in pairs:
        assert feat_project(seq_a, seq_b) == feat_library(seq_a, seq_b), (seq_a, seq_b)
