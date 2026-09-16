"""PyMC adapter for the Kahneman & Tversky local-representativeness family.

This quantitative operationalization separates K&T's local-balance and
irregularity claims. Balance is averaged over the whole sequence and sliding
windows at scales two through four. Irregularity combines distance from an
over-alternating prototype with an explicit periodic-template penalty. See the
pure-Python twin in
``model_families/local_representativeness.py`` for the full rationale.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    multiscale_imbalance_a = pm.Data(
        "multiscale_imbalance_a", np.zeros(1, dtype="float64")
    )
    multiscale_imbalance_b = pm.Data(
        "multiscale_imbalance_b", np.zeros(1, dtype="float64")
    )
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.5001, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    periodic_share = pm.Uniform("periodic_share", lower=0.01, upper=0.99)
    beta = pm.Uniform("beta", lower=0.2, upper=12.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    balance_weight = 1.0 - alt_weight
    irregularity_a = (
        (1.0 - periodic_share) * pt.abs(p_alts_a - theta_alt)
        + periodic_share * periodicity_a
    )
    irregularity_b = (
        (1.0 - periodic_share) * pt.abs(p_alts_b - theta_alt)
        + periodic_share * periodicity_b
    )
    score_a = -(
        balance_weight * multiscale_imbalance_a + alt_weight * irregularity_a
    )
    score_b = -(
        balance_weight * multiscale_imbalance_b + alt_weight * irregularity_b
    )

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)


# ─────────────────────────────────────────────────────────────────────────────
# Raw-features (arm C) support: this model computes the columns it binds
# ─────────────────────────────────────────────────────────────────────────────
# In a `raw_features` run the agents' responses CSV carries only the raw H/T
# sequences, so no model may depend on harness-provided feature columns (see
# docs/raw_features_arm.md). This model therefore derives the columns its
# `pm.Data` containers read from the sequences themselves, through the
# `compute_features(sequence_a, sequence_b)` hook that
# `src/models/pymc_inference.py` applies wherever rows become model data —
# fitting, the EIG design and the held-out prediction path alike.
#
# The helpers below are copied VERBATIM from
# `src/subjective_randomness/features.py` (extracted programmatically, not
# retyped) so this file is self-contained: it imports no project featurizer,
# and it leaves nothing for a candidate agent to import either.
# `tests/test_raw_features_seeds.py` pins each copy to its original and pins
# `compute_features` to the featurizer's values for these columns, so the
# copies cannot silently drift.
#
# Column collision is deliberate and fatal: the hook raises if a name it
# returns is already present, so these models must NOT be fitted on a
# featurized CSV. That is what makes a raw-features run all-or-nothing.

LOCAL_WINDOW = 4


def clean_sequence(seq: str) -> str:
    """Uppercase an H/T sequence and reject empty input.

    An empty sequence is never a legitimate trial — it means upstream breakage
    (a stimulus without ``sequence_a``, a truncated responses.csv) — so every
    helper below raises rather than emitting a zero-filled feature row that
    reads like a real observation. The model families' ``clean_sequence``
    (``model_families/common.py``) makes the same call and additionally rejects
    non-H/T symbols; this module keeps its own copy so it stays importable
    without the model-family package.
    """
    s = seq.strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def periodicity_score(seq: str) -> float:
    """Degree to which a sequence matches a short repeating template.

    The model-family helper of the same name in ``model_families/common.py``
    wraps this one.
    """
    s = clean_sequence(seq)
    n = len(s)
    if n <= 2:
        return 0.0
    best_match = 0.5
    for period in range(1, (n // 2) + 1):
        template = s[:period]
        matches = sum(1 for i, c in enumerate(s) if c == template[i % period])
        best_match = max(best_match, matches / n)
    return max(0.0, min(1.0, 2.0 * (best_match - 0.5)))


def multiscale_local_imbalance(seq: str) -> float:
    """Mean H/T imbalance across global and short local descriptions.

    The global sequence and each sliding-window scale from two through four
    receive equal weight. Within a scale, every window receives equal weight.
    This makes the operationalization explicit and avoids allowing a single
    worst window to determine the entire score. The model-family helper of the
    same name in ``model_families/common.py`` wraps this one.
    """
    s = clean_sequence(seq)
    n = len(s)
    heads = sum(1 for c in s if c == "H")
    global_imbalance = 2.0 * abs(heads / n - 0.5)
    scale_scores = [global_imbalance]
    for window in range(2, min(LOCAL_WINDOW, n - 1) + 1):
        window_scores = []
        for start in range(n - window + 1):
            chunk = s[start : start + window]
            chunk_heads = sum(1 for c in chunk if c == "H")
            window_scores.append(2.0 * abs(chunk_heads / window - 0.5))
        scale_scores.append(sum(window_scores) / len(window_scores))
    return sum(scale_scores) / len(scale_scores)


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """`p_alts_{a,b}`, `periodicity_{a,b}` and `multiscale_imbalance_{a,b}`.

    The `n > 0` guard on the multiscale term mirrors `sequence_features_float`
    exactly, even though `clean_sequence` already rejects empty input.
    """
    features: dict = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        cleaned = clean_sequence(seq)
        n = len(cleaned)
        alts = sum(1 for i in range(1, n) if cleaned[i] != cleaned[i - 1])
        features[f"p_alts_{suffix}"] = (alts / (n - 1)) if n > 1 else 0.0
        features[f"periodicity_{suffix}"] = periodicity_score(seq)
        features[f"multiscale_imbalance_{suffix}"] = (
            multiscale_local_imbalance(seq) if n > 0 else 0.0
        )
    return features
