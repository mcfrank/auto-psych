"""People judge randomness by multi-scale representativeness: they expect the proportion of heads to closely match the 50/50 fair-coin ideal not just globally, but across all possible contiguous sub-sequences. Rather than using separate heuristics for global imbalance and alternations, people penalize a sequence based on the mean squared deviation of the head proportion from 0.5 across all sub-windows of length 2 or greater, which naturally favors evenly spaced, periodic sequences because those exhibit the least local variance."""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Compute the multi-scale representativeness penalty."""

    def multiscale_deviation(seq):
        mses = []
        n = len(seq)
        for w in range(2, n + 1):
            for i in range(n - w + 1):
                prop_h = seq[i : i + w].count("H") / w
                mses.append((prop_h - 0.5) ** 2)
        return float(np.mean(mses)) if mses else 0.0

    return {
        "multiscale_dev_a": multiscale_deviation(sequence_a),
        "multiscale_dev_b": multiscale_deviation(sequence_b),
    }


with pm.Model() as model:
    # Stimulus inputs
    multiscale_dev_a = pm.Data("multiscale_dev_a", np.zeros(1, dtype="float64"))
    multiscale_dev_b = pm.Data("multiscale_dev_b", np.zeros(1, dtype="float64"))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    beta = pm.Uniform("beta", lower=0.1, upper=50.0)
    side_bias = pm.Uniform("side_bias", lower=-2.0, upper=2.0)

    # Calculate scores (negative penalty)
    score_a = -multiscale_dev_a
    score_b = -multiscale_dev_b

    # Choice probability with a psychometric function
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )

    # Likelihood
    pm.Bernoulli("response", p=p_left, observed=chose_left)
