"""People judge a sequence as more random when its run-length encoding (RLE) requires
more blocks: the number of RLE blocks equals (alternations + 1), and more blocks per
character means the sequence resists RLE compression, making it appear more random.
Unlike prototype-similarity, this predicts a monotonic relationship — more alternations
always means more random, with no learned ideal alternation rate."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


with pm.Model() as model:
    alts_a = pm.Data("alts_a", np.zeros(1, dtype="int64"))
    alts_b = pm.Data("alts_b", np.zeros(1, dtype="int64"))
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.HalfNormal("beta", sigma=5.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # RLE blocks = (alternations + 1); normalize by length to compare sequences of
    # different lengths on equal footing
    rle_a = pt.cast(alts_a + 1, "float64") / pt.cast(pt.maximum(n_a, 1), "float64")
    rle_b = pt.cast(alts_b + 1, "float64") / pt.cast(pt.maximum(n_b, 1), "float64")

    # More RLE blocks per character → harder to compress → more random-looking
    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta * (rle_a - rle_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
