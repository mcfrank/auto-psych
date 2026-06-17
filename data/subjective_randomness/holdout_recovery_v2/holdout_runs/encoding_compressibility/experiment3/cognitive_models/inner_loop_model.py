"""People judge a sequence as more random when its head proportion is closer to 0.5 —
they assess perceived randomness solely by whether the sequence looks like it came from
a fair (unbiased) coin, ignoring alternation patterns and run structure entirely."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


with pm.Model() as model:
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Less imbalance → closer to 50/50 → more random-looking
    score_a = -imbalance_a
    score_b = -imbalance_b

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
