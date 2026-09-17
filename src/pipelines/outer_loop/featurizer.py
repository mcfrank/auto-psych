"""Raw response column constants for the pipeline.

Every model computes its own features from raw stimulus rows via its
``compute_features`` or ``prepare_observed`` hook. The pipeline passes only the
raw columns below; the featurizer infrastructure that used to live here was
removed when raw became the only mode.
"""

from __future__ import annotations

RAW_RESPONSE_COLUMNS = (
    "sequence_a",
    "sequence_b",
    "participant_id",
    "trial_index",
    "chose_left",
)
