"""Featurize subjective_randomness responses for PyMC models (project asset).

This is the per-project featurizer the pipeline loads **by file path**:
`_load_project_featurizer` (inner loop) and the EIG `--featurize` flag both read
`featurize_stimulus` from this file. To keep the project asset and the standalone
research library from silently drifting, the implementation lives in exactly one
place — `src.subjective_randomness.features` — and this module re-exports it.
(`tests/test_featurizer_parity.py` pins the two to identical output; the
model-recovery loop generates synthetic responses via `features.py` while the
human experiment featurizes via this file, so any divergence would make the two
loops incomparable.)

Feature columns per sequence (`a` and `b`):
    n_<x>             total length                        (int)
    h_<x>             head count                          (int)
    alts_<x>          alternation count (H/T transitions) (int)
    max_run_<x>       longest constant run                (int)
    rep_motifs_<x>    repetition motifs in motif parse    (int)
    alt_motifs_<x>    alternation motifs in motif parse   (int)
    p_<x>             head proportion                     (float)
    p_alts_<x>        alternation proportion              (float)
    max_run_norm_<x>  longest run scaled to [0, 1]        (float)
    imbalance_<x>     distance from 50/50 heads/tails     (float)
    periodicity_<x>   short repeating-template score       (float)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Loaded by path (importlib.spec_from_file_location), so `src` is not guaranteed
# to be importable yet — put the repo root on sys.path before importing the
# canonical featurizer. REPO_ROOT is five parents up:
# src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.subjective_randomness.features import featurize_stimulus  # noqa: E402

__all__ = ["featurize_stimulus"]
