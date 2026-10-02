"""Signed personal decisiveness.

People differ in how they act on their impression of randomness along a single
signed dimension: most choose the sequence that looks more random with a
personal strength, some barely follow their impression, and a few systematically
pick the sequence that looks less random (reading the task in reverse). This
one personal signed decisiveness (population mean fixed at 1, so the shared cue
weights set the scale) replaces separate personal guessing rates and personal
switching-rate sensitivities, and it alone carries the person-to-person spread
in how strongly the cues matter; each sequence's impression is otherwise the
current best judgement (personal ideal switching rate with an asymmetric
tolerance, shared weights on imbalance, the longest run, the final run, mirror symmetry, run-length variety, the most lopsided stretch
and complement symmetry).
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

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Shared alternation sensitivity.
    log_beta = pm.Normal("log_beta", mu=2.5, sigma=0.7)
    beta = pm.math.exp(log_beta)

    # Shared weights on H/T imbalance and the longest run (shared, not personal:
    # personal weights would trade off against the personal signed decisiveness).
    g = pm.Normal("gamma", mu=0.0, sigma=2.0)
    d = pm.Normal("delta", mu=0.0, sigma=2.0)

    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    rho = pm.Normal("rho", mu=0.0, sigma=1.5)
    omega = pm.Normal("omega", mu=0.0, sigma=1.5)
    eta = pm.Normal("eta", mu=0.0, sigma=1.5)
    chi = pm.Normal("chi", mu=0.0, sigma=1.5)

    # Shared fraction of the ideal-rate penalty applied to over-alternation.
    log_over = pm.Normal("log_over", mu=0.0, sigma=1.0)
    over_frac = pm.Deterministic("over_frac", pm.math.exp(log_over))

    # Personal signed decisiveness: population mean 1, free to cross zero.
    sigma_s = pm.HalfNormal("sigma_s", sigma=0.7)
    z_s = pm.Normal("z_s", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    decisiveness = pm.Deterministic("decisiveness", 1.0 + sigma_s * z_s)

    theta = ideal[participant_id]

    def rate_penalty(rate):
        dev = rate - theta
        return pt.sqr(dev) * pt.switch(dev > 0, over_frac, 1.0)

    score_a = -beta * rate_penalty(alt_rate_a) - g * imbalance_a - d * maxrun_a - kappa * endrun_a - rho * palin_a + omega * variety_a - eta * lopsided_a - chi * antisym_a
    score_b = -beta * rate_penalty(alt_rate_b) - g * imbalance_b - d * maxrun_b - kappa * endrun_b - rho * palin_b + omega * variety_b - eta * lopsided_b - chi * antisym_b
    s = decisiveness[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(s * (score_a - score_b)), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
