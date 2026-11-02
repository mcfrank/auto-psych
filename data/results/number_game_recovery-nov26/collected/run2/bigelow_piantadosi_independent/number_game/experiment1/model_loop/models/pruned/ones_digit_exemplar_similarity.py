"""Ones-digit exemplar similarity.

People judge a target by comparing its final (ones) digit with the final digit
of each example, ignoring magnitude and mathematical rules: a target is endorsed
in proportion to how many examples it resembles in surface form, where an
example with the identical final digit counts fully and an example whose final
digit merely has the same parity (odd/even) counts partially.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    """Fraction of examples with the same ones digit, and with only the same parity."""
    examples = [int(x) for x in number_set.split(",")]
    t = int(target) % 10
    same_digit = sum(1 for x in examples if x % 10 == t)
    same_parity_only = sum(1 for x in examples if x % 10 != t and (x % 2) == (t % 2))
    n = float(len(examples))
    return {
        "frac_same_digit": same_digit / n,
        "frac_parity_only": same_parity_only / n,
    }


with pm.Model() as model:
    frac_same_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_parity_only = pm.Data("frac_parity_only", np.zeros(1, dtype="float64"))

    # Baseline log-odds of yes for a target resembling no example.
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    # How strongly summed digit similarity drives endorsement.
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Partial credit for an example sharing only parity of the final digit.
    parity_weight = pm.Beta("parity_weight", alpha=2.0, beta=2.0)

    similarity = frac_same_digit + parity_weight * frac_parity_only
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * similarity))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
