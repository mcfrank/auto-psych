"""The number-game run's final, even_rule_digit_size_principle_counting, with switchable mechanisms.

A logistic model of P(yes) on features of the example set and the target.
Every term of its logit is a mechanism, the intercept included; a model keeps
at least one (``KEEP_ONE_OF``): with none, P(yes) is 0.5 on every item and
there is nothing to fit.

The mechanisms (``MECHANISMS``), in the order of the logit:

* ``intercept``: a bias toward yes or no.
* ``similarity`` (gain): credit for the target's final-digit similarity to the
  examples (a shared last digit, plus parity credit below).
* ``parity`` (s_parity): partial similarity for a shared parity without a
  shared last digit, in the examples' similarity to the target and to each
  other (off: last digits only).
* ``parity_preempt`` (preempt): a shared final-digit rule pre-empts the parity
  credit.
* ``mutual_standard`` (standard): a penalty for the shortfall of the target's
  similarity below the examples' similarity to each other.
* ``common_factor`` (factor_boost): multiples of the examples' common factor
  (>= 3, not a multiple of 5).
* ``round_factor`` (round_boost): multiples of a round common factor (a
  multiple of 5).
* ``decade_band`` (band_boost, band_width): familiarity for targets within or
  near the examples' decade band.
* ``magnitude_preempt`` (magnitude_preempt): an explicit rule (a shared last
  digit or a common factor >= 3) pre-empts the decade-band credit.
* ``counting`` (count_boost): support for targets reached by counting up from
  one example in steps of its size.
* ``count_step_cost`` (step_cost): that support decays with each extra step
  (off: every reach counts alike).
* ``size_principle`` (size_strength): each confirming example beyond the second
  makes breaking a shared final-digit rule cost more.
* ``even_rule`` (even_rule_penalty): an all-even set rejects odd targets by an
  extra amount ("even numbers" is a privileged concept, "odd numbers" not).

``parity`` acts only through the similarity and the mutual standard, so it
needs one of them; the two pre-emptions and the step cost need the term they
modify.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

FINAL = "even_rule_digit_size_principle_counting"
MECHANISMS = (
    "intercept", "similarity", "parity", "parity_preempt", "mutual_standard", "common_factor", "round_factor",
    "decade_band", "magnitude_preempt", "counting", "count_step_cost", "size_principle", "even_rule",
)
REQUIRES = {
    "parity": ("similarity", "mutual_standard"),
    "parity_preempt": ("parity",),
    "magnitude_preempt": ("decade_band",),
    "count_step_cost": ("counting",),
}
KEEP_ONE_OF = (MECHANISMS,)
# For variant names (``variants.variant_name``).
SHORT = {
    "intercept": "int", "similarity": "sim", "parity": "par", "parity_preempt": "parpre", "mutual_standard": "std",
    "common_factor": "cf", "round_factor": "round", "decade_band": "band", "magnitude_preempt": "magpre",
    "counting": "count", "count_step_cost": "step", "size_principle": "size", "even_rule": "even",
}
NEUTRAL = {
    "intercept": {"intercept": 0.0},
    "similarity": {"gain": 0.0},
    "parity": {"s_parity": 0.0},
    "parity_preempt": {"preempt": 0.0},
    "mutual_standard": {"standard": 0.0},
    "common_factor": {"factor_boost": 0.0},
    "round_factor": {"round_boost": 0.0},
    "decade_band": {"band_boost": 0.0},
    "magnitude_preempt": {"magnitude_preempt": 0.0},
    "counting": {"count_boost": 0.0},
    "count_step_cost": {"step_cost": 0.0},
    "size_principle": {"size_strength": 0.0},
    "even_rule": {"even_rule_penalty": 0.0},
}


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    g = reduce(math.gcd, xs)
    lo = min(x // 10 for x in xs)
    hi = max(x // 10 for x in xs)
    td = t // 10
    uniform = len(xs) >= 2 and len({x % 10 for x in xs}) == 1
    pairs = [(a, b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    if pairs:
        pair_digit = sum(1 for a, b in pairs if a % 10 == b % 10) / len(pairs)
        pair_parity = sum(
            1 for a, b in pairs if a % 2 == b % 2 and a % 10 != b % 10
        ) / len(pairs)
    else:
        pair_digit = pair_parity = 0.0
    # Counting up from a single example: the fewest steps (multiplier k >= 2)
    # at which some example of 3 or more reaches the target.
    steps = [t // x for x in xs if x >= 3 and t != x and t % x == 0]
    all_even = all(x % 2 == 0 for x in xs)
    return {
        "frac_same_digit": same_digit,
        "frac_same_parity_other": same_parity_other,
        "common_factor_multiple": 1.0 if (g >= 3 and g % 5 != 0 and t % g == 0) else 0.0,
        "round_factor_multiple": 1.0 if (g % 5 == 0 and t % g == 0) else 0.0,
        "decade_band_gap": float(max(lo - td, 0, td - hi)),
        "shared_digit_rule": 1.0 if uniform else 0.0,
        "explicit_rule_detected": 1.0 if (len(xs) >= 2 and (uniform or g >= 3)) else 0.0,
        "pair_same_digit": pair_digit,
        "pair_same_parity_other": pair_parity,
        "reached_by_counting": 1.0 if steps else 0.0,
        "extra_count_steps": float(min(steps) - 2) if steps else 0.0,
        # The target violates an "even numbers" rule the all-even set evokes.
        "breaks_even_rule": 1.0 if (all_even and t % 2 == 1) else 0.0,
        # Confirming examples beyond the second for a broken shared-digit rule.
        "digit_rule_violation_evidence": (
            float(len(xs) - 2) if (uniform and t % 10 != xs[0] % 10) else 0.0
        ),
    }


def build(on):
    on = frozenset(on)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")

    with pm.Model() as model:
        frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
        frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
        cf_mult = pm.Data("common_factor_multiple", np.zeros(1, dtype="float64"))
        band_gap = pm.Data("decade_band_gap", np.zeros(1, dtype="float64"))
        rule = pm.Data("shared_digit_rule", np.zeros(1, dtype="float64"))
        any_rule = pm.Data("explicit_rule_detected", np.zeros(1, dtype="float64"))
        round_mult = pm.Data("round_factor_multiple", np.zeros(1, dtype="float64"))
        pair_digit = pm.Data("pair_same_digit", np.zeros(1, dtype="float64"))
        pair_par = pm.Data("pair_same_parity_other", np.zeros(1, dtype="float64"))
        reached = pm.Data("reached_by_counting", np.zeros(1, dtype="float64"))
        extra_steps = pm.Data("extra_count_steps", np.zeros(1, dtype="float64"))
        breaks_even = pm.Data("breaks_even_rule", np.zeros(1, dtype="float64"))
        violation_evidence = pm.Data("digit_rule_violation_evidence", np.zeros(1, dtype="float64"))

        s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0) if "parity" in on else 0.0
        preempt = pm.Beta("preempt", alpha=2.0, beta=2.0) if "parity_preempt" in on else 0.0
        mean_sim = frac_digit + s_parity * (1.0 - preempt * rule) * frac_par
        mutual_sim = pair_digit + s_parity * pair_par
        shortfall = pt.softplus(20.0 * (mutual_sim - mean_sim)) / 20.0

        intercept = pm.Normal("intercept", mu=0.0, sigma=2.0) if "intercept" in on else 0.0
        gain = pm.HalfNormal("gain", sigma=5.0) if "similarity" in on else 0.0
        standard = pm.HalfNormal("standard", sigma=3.0) if "mutual_standard" in on else 0.0
        factor_boost = pm.Normal("factor_boost", mu=0.0, sigma=2.0) if "common_factor" in on else 0.0
        round_boost = pm.Normal("round_boost", mu=0.0, sigma=2.0) if "round_factor" in on else 0.0
        if "decade_band" in on:
            band_boost = pm.HalfNormal("band_boost", sigma=2.0)
            band_width = pm.LogNormal("band_width", mu=np.log(1.0), sigma=0.75)
            band_sim = pm.math.exp(-band_gap / band_width)
            magnitude_preempt = (pm.Beta("magnitude_preempt", alpha=2.0, beta=2.0)
                                 if "magnitude_preempt" in on else 0.0)
            band_term = band_boost * (1.0 - magnitude_preempt * any_rule) * band_sim
        else:
            band_term = 0.0

        # Support from counting up from one example, decaying per extra step.
        if "counting" in on:
            count_boost = pm.HalfNormal("count_boost", sigma=1.5)
            step_cost = pm.HalfNormal("step_cost", sigma=1.0) if "count_step_cost" in on else 0.0
            counting_support = count_boost * reached * pm.math.exp(-step_cost * extra_steps)
        else:
            counting_support = 0.0

        # Size principle: per extra confirming example, a broken digit rule costs more.
        size_strength = pm.HalfNormal("size_strength", sigma=1.5) if "size_principle" in on else 0.0

        # Extra rejection of odd targets once all examples are even.
        even_rule_penalty = pm.HalfNormal("even_rule_penalty", sigma=2.0) if "even_rule" in on else 0.0

        p_yes = pm.Deterministic(
            "p_yes",
            pm.math.sigmoid(
                intercept + gain * mean_sim
                - standard * shortfall
                + factor_boost * cf_mult + round_boost * round_mult
                + band_term
                + counting_support
                - size_strength * violation_evidence
                - even_rule_penalty * breaks_even
            ),
        )

        rating = pm.Data("rating", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=p_yes, observed=rating)
    return model
