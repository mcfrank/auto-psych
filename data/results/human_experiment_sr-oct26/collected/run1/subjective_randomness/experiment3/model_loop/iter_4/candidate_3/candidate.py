"""Personal engagement factor.

Refinement of personal_pattern_detection_gain. Each sequence is judged by its
switching rate against a personal ideal (personal sensitivity, asymmetric
tolerance), personal weights on imbalance and the longest run, and shared
structural cues of design scaled by a personal pattern-detection gain, with a
personal lapse rate. The one change: a single personal engagement level
jointly raises alternation sensitivity and pattern-detection gain and lowers
the lapse rate (fitted loadings), so a person's consistency traits are
correlated rather than independent.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def imbalance(seq):
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    def longest_run_share(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return (best - 1) / (len(seq) - 1)

    def final_run_share(seq):
        cur = 1
        for x, y in zip(seq[-2::-1], seq[:0:-1]):
            if x != y:
                break
            cur += 1
        return (cur - 1) / (len(seq) - 1)

    def palindrome(seq):
        return 1.0 if seq == seq[::-1] else 0.0

    def run_variety(seq):
        runs = []
        cur = 1
        for x, y in zip(seq, seq[1:]):
            if x == y:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        n = len(seq)
        kmax = 1
        while (kmax + 1) * (kmax + 2) // 2 <= n:
            kmax += 1
        if kmax <= 1:
            return 0.0
        return (len(set(runs)) - 1) / (kmax - 1)

    def complement_symmetric(seq):
        if len(seq) < 4:
            return 0.0
        flipped = "".join("T" if c == "H" else "H" for c in seq[::-1])
        return 1.0 if seq == flipped else 0.0

    def lopsided_stretch(seq):
        # Largest |#H - #T| over contiguous stretches = range of the running tally.
        tally, lo, hi = 0, 0, 0
        for x in seq:
            tally += 1 if x == "H" else -1
            lo = min(lo, tally)
            hi = max(hi, tally)
        return (hi - lo - 1) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "maxrun_a": longest_run_share(a),
        "maxrun_b": longest_run_share(b),
        "endrun_a": final_run_share(a),
        "endrun_b": final_run_share(b),
        "palin_a": palindrome(a),
        "palin_b": palindrome(b),
        "variety_a": run_variety(a),
        "variety_b": run_variety(b),
        "lopsided_a": lopsided_stretch(a),
        "lopsided_b": lopsided_stretch(b),
        "antisym_a": complement_symmetric(a),
        "antisym_b": complement_symmetric(b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    maxrun_a = pm.Data("maxrun_a", np.zeros(1, dtype="float64"))
    maxrun_b = pm.Data("maxrun_b", np.zeros(1, dtype="float64"))
    endrun_a = pm.Data("endrun_a", np.zeros(1, dtype="float64"))
    endrun_b = pm.Data("endrun_b", np.zeros(1, dtype="float64"))
    palin_a = pm.Data("palin_a", np.zeros(1, dtype="float64"))
    palin_b = pm.Data("palin_b", np.zeros(1, dtype="float64"))
    variety_a = pm.Data("variety_a", np.zeros(1, dtype="float64"))
    variety_b = pm.Data("variety_b", np.zeros(1, dtype="float64"))
    lopsided_a = pm.Data("lopsided_a", np.zeros(1, dtype="float64"))
    lopsided_b = pm.Data("lopsided_b", np.zeros(1, dtype="float64"))
    antisym_a = pm.Data("antisym_a", np.zeros(1, dtype="float64"))
    antisym_b = pm.Data("antisym_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal engagement level shared by the three consistency traits.
    z_eng = pm.Normal("z_eng", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    load_beta = pm.HalfNormal("load_beta", sigma=0.5)
    load_gain = pm.Normal("load_gain", mu=0.0, sigma=0.5)
    load_lapse = pm.Normal("load_lapse", mu=0.0, sigma=0.7)

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    # Personal alternation sensitivity (non-centred log-normal population).
    mu_beta = pm.Normal("mu_beta", mu=2.5, sigma=0.5)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    beta = pm.math.exp(mu_beta + sigma_beta * z_beta + load_beta * z_eng)

    # Personal weight on H/T imbalance.
    mu_gamma = pm.Normal("mu_gamma", mu=0.0, sigma=2.0)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.5)
    z_gamma = pm.Normal("z_gamma", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gamma = mu_gamma + sigma_gamma * z_gamma

    # Personal streak aversion (weight on the longest run's share of the sequence).
    mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.5)
    z_delta = pm.Normal("z_delta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    delta = mu_delta + sigma_delta * z_delta

    # Shared extra weight on the final run (positive: end streaks penalised more).
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)

    # Shared penalty on mirror symmetry (positive: palindromes look designed).
    rho = pm.Normal("rho", mu=0.0, sigma=1.5)

    # Shared weight on run-length variety (positive: varied runs look random).
    omega = pm.Normal("omega", mu=0.0, sigma=1.5)

    # Shared penalty on the most lopsided stretch (positive: local excess looks designed).
    eta = pm.Normal("eta", mu=0.0, sigma=1.5)

    # Shared penalty on complement symmetry (positive: antisymmetric sequences look designed).
    chi = pm.Normal("chi", mu=0.0, sigma=1.5)

    # Personal pattern-detection gain scaling all structural cues (median 1).
    sigma_gain = pm.HalfNormal("sigma_gain", sigma=0.7)
    z_gain = pm.Normal("z_gain", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gain = pm.math.exp(sigma_gain * z_gain + load_gain * z_eng)

    # Shared fraction of the ideal-rate penalty applied to over-alternation.
    log_over = pm.Normal("log_over", mu=0.0, sigma=1.0)
    over_frac = pm.Deterministic("over_frac", pm.math.exp(log_over))

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse - load_lapse * z_eng))

    theta = ideal[participant_id]
    g = gamma[participant_id]
    d = delta[participant_id]
    b = beta[participant_id]
    s = gain[participant_id]

    def structure(endrun, palin, variety, lopsided, antisym):
        return -kappa * endrun - rho * palin + omega * variety - eta * lopsided - chi * antisym

    def rate_penalty(rate):
        dev = rate - theta
        return pt.sqr(dev) * pt.switch(dev > 0, over_frac, 1.0)

    score_a = -b * rate_penalty(alt_rate_a) - g * imbalance_a - d * maxrun_a + s * structure(endrun_a, palin_a, variety_a, lopsided_a, antisym_a)
    score_b = -b * rate_penalty(alt_rate_b) - g * imbalance_b - d * maxrun_b + s * structure(endrun_b, palin_b, variety_b, lopsided_b, antisym_b)
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
