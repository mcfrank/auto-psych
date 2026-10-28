"""Place-value digit-set induction.

People do not think about these numbers arithmetically or as magnitudes on a
number line; they treat each number as a two-digit written string (tens digit,
final digit; 100 is read as "00") and infer which tens digits and which final
digits the program may write. Each digit is a priori allowed with an inferred
probability (one for tens digits, one for final digits), and examples are
treated as random draws from the allowed tens-by-final-digit grid, so smaller
digit sets that cover the examples are favoured (a tempered size principle).
A target is endorsed to the degree that both its tens digit and its final
digit are believed allowed, passed through a decision gain and criterion. So
after 2, 8 people accept 62 or 88 but reject 16, and after 64, 4 they accept
14, 24, 94 but not 16: generalization follows shared digits, not powers or
multiples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

K = np.arange(1, 11, dtype="float64")  # possible number of allowed digits


def _digits(n):
    n = n % 100
    return n // 10, n % 10


def _pattern(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    tens = {_digits(x)[0] for x in ex}
    units = {_digits(x)[1] for x in ex}
    tt, tu = _digits(int(target))
    return (len(tens), len(units), len(ex), int(tt in tens), int(tu in units))


def prepare_observed(rows):
    """Collapse trials to distinct (digit counts, set size, target-digit-seen) patterns."""
    pats = [_pattern(r["number_set"], r["target"]) for r in rows]
    unique = list(dict.fromkeys(pats))
    index = {p: i for i, p in enumerate(unique)}
    u = np.array(unique, dtype="float64")
    return {
        "n_tens_seen": u[:, 0],
        "n_units_seen": u[:, 1],
        "n_examples": u[:, 2],
        "tens_seen": u[:, 3],
        "units_seen": u[:, 4],
        "row_pattern": np.array([index[p] for p in pats], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


def _log_choose(n, k):
    return pt.gammaln(n + 1) - pt.gammaln(k + 1) - pt.gammaln(n - k + 1)


with pm.Model() as model:
    a = pm.Data("n_tens_seen", np.ones(1, dtype="float64"))
    b = pm.Data("n_units_seen", np.ones(1, dtype="float64"))
    n_ex = pm.Data("n_examples", np.ones(1, dtype="float64"))
    tens_seen = pm.Data("tens_seen", np.zeros(1, dtype="float64"))
    units_seen = pm.Data("units_seen", np.zeros(1, dtype="float64"))
    row_pattern = pm.Data("row_pattern", np.zeros(1, dtype="int64"))

    logit_q_tens = pm.Normal("logit_q_tens", mu=0.0, sigma=1.5)
    logit_q_units = pm.Normal("logit_q_units", mu=0.0, sigma=1.5)
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    k = pt.as_tensor(K)

    def digit_prior(seen, logit_q):
        # log P(allowed-set size = k | seen digits included), unnormalised
        s = seen[:, None]
        kk = k[None, :]
        valid = kk >= s
        extra = pt.maximum(kk - s, 0.0)
        lp = _log_choose(10.0 - s, pt.minimum(extra, 10.0 - s)) + kk * -pt.softplus(-logit_q) + (
            10.0 - kk
        ) * -pt.softplus(logit_q)
        lp = pt.switch(valid, lp, -1e4)
        # probability an unseen digit is allowed, given k allowed digits
        frac = pt.switch(s < 10.0, extra / pt.maximum(10.0 - s, 1.0), 1.0)
        return lp, frac

    lp_t, frac_t = digit_prior(a, logit_q_tens)  # (trials, 10)
    lp_u, frac_u = digit_prior(b, logit_q_units)
    log_size = pt.log(k)[None, :, None] + pt.log(k)[None, None, :]
    log_post = lp_t[:, :, None] + lp_u[:, None, :] - evidence_weight * n_ex[:, None, None] * log_size
    post = pt.exp(log_post - pt.logsumexp(log_post, axis=(1, 2), keepdims=True))
    in_t = tens_seen[:, None] + (1.0 - tens_seen[:, None]) * frac_t  # (trials, 10)
    in_u = units_seen[:, None] + (1.0 - units_seen[:, None]) * frac_u
    belief = (post * in_t[:, :, None] * in_u[:, None, :]).sum(axis=(1, 2))
    belief = pt.clip(belief, 1e-6, 1 - 1e-6)[row_pattern]

    belief_logit = pt.log(belief) - pt.log1p(-belief)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
