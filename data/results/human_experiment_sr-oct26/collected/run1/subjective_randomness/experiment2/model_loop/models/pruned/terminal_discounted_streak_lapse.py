"""Personal ideal alternation, balance weight, streak aversion and lapse, with the final run weighed apart.

Refinement of `individual_streak_aversion_lapse`: each person judges a sequence
as more random the closer its alternation rate lies to their own ideal switching
rate, penalises heads/tails imbalance and the most striking streak by weights of
their own, and guesses on some trials at a personal lapse rate. The one change:
the run a sequence ends on (still in progress when reading stops) has its length
scaled by a fitted shared factor before the most striking streak is chosen, so a
final streak can count less (or more) than an equally long closed-off one.
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

    def runs(seq):
        out = [1]
        for x, y in zip(seq, seq[1:]):
            if x == y:
                out[-1] += 1
            else:
                out.append(1)
        return out

    def closed_and_final(seq):
        r = runs(seq)
        n1 = len(seq) - 1
        closed = max(r[:-1]) if len(r) > 1 else 1
        return (closed - 1) / n1, (r[-1] - 1) / n1

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    ca, fa = closed_and_final(a)
    cb, fb = closed_and_final(b)
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "closed_run_a": ca,
        "closed_run_b": cb,
        "final_run_a": fa,
        "final_run_b": fb,
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    closed_run_a = pm.Data("closed_run_a", np.zeros(1, dtype="float64"))
    closed_run_b = pm.Data("closed_run_b", np.zeros(1, dtype="float64"))
    final_run_a = pm.Data("final_run_a", np.zeros(1, dtype="float64"))
    final_run_b = pm.Data("final_run_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.5, sigma=0.5)

    mu_gamma = pm.Normal("mu_gamma", mu=0.0, sigma=2.0)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.5)
    z_gamma = pm.Normal("z_gamma", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gamma = mu_gamma + sigma_gamma * z_gamma

    mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.5)
    z_delta = pm.Normal("z_delta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    delta = mu_delta + sigma_delta * z_delta

    # Shared weight on the final (unfinished) run relative to a closed-off run.
    log_final_weight = pm.Normal("log_final_weight", mu=0.0, sigma=0.5)
    final_weight = pm.Deterministic("final_weight", pt.exp(log_final_weight))

    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    streak_a = pt.maximum(closed_run_a, final_weight * final_run_a)
    streak_b = pt.maximum(closed_run_b, final_weight * final_run_b)

    theta = ideal[participant_id]
    g = gamma[participant_id]
    d = delta[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - g * imbalance_a - d * streak_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - g * imbalance_b - d * streak_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
