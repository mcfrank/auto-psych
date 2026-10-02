"""Bayesian generator inference with a personal prior suspicion of non-randomness.

People judge randomness normatively: for each sequence they compute the
posterior probability that it came from a fair coin rather than from a
non-random generator (a biased coin, a sticky-or-switchy Markov coin, or a
mirror- or complement-symmetric construction), and pick the sequence with the
higher posterior probability of being random. The one distortion is that each
person brings their own prior belief that a sequence was made by a non-random
generator, so the posterior saturates differently for different people: the
suspicious discriminate only among random-looking sequences, the trusting only
among regular-looking ones. People also guess on a small shared share of trials.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400
# Typical log Bayes factor of a stimulus sequence: the decision sensitivity is
# expressed per unit of log evidence at this reference, so it does not trade
# off against how far a person's prior suspicion saturates the posterior.
BF_REF = -0.4


def compute_features(sequence_a, sequence_b):
    def log_bayes_factor(seq):
        """log P(seq | fair coin) - log P(seq | non-random generators, equal prior)."""
        n = len(seq)
        if n < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        log_fair = -n * math.log(2.0)
        h = seq.count("H")
        t = n - h
        # Biased coin, uniform prior on P(heads).
        log_biased = math.lgamma(h + 1) + math.lgamma(t + 1) - math.lgamma(n + 2)
        # Markov coin: first flip fair, uniform prior on switch probability.
        s = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        r = (n - 1) - s
        log_markov = -math.log(2.0) + math.lgamma(s + 1) + math.lgamma(r + 1) - math.lgamma(n + 1)
        alts = [log_biased, log_markov]
        # Mirror symmetry: first half fair, second half its reflection.
        half = (n + 1) // 2
        if n >= 4 and seq == seq[::-1]:
            alts.append(-half * math.log(2.0))
        else:
            alts.append(-math.inf)
        # Complement symmetry: second half is the reversed, H/T-swapped first half.
        flipped = "".join("T" if c == "H" else "H" for c in seq[::-1])
        if n >= 4 and seq == flipped:
            alts.append(-(n // 2) * math.log(2.0))
        else:
            alts.append(-math.inf)
        m = max(alts)
        log_alt = m + math.log(sum(math.exp(a - m) for a in alts if a > -math.inf)) - math.log(len(alts))
        return log_fair - log_alt

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {"log_bf_a": log_bayes_factor(a), "log_bf_b": log_bayes_factor(b)}


with pm.Model() as model:
    log_bf_a = pm.Data("log_bf_a", np.zeros(1, dtype="float64"))
    log_bf_b = pm.Data("log_bf_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal prior log-odds that a sequence came from a non-random generator
    # (non-centred, weakly informative on the scale of the evidence).
    mu_susp = pm.Normal("mu_susp", mu=0.0, sigma=1.0)
    sigma_susp = pm.HalfNormal("sigma_susp", sigma=0.5)
    z_susp = pm.Normal("z_susp", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    suspicion = mu_susp + sigma_susp * z_susp

    # Shared decisiveness per unit of log evidence at the reference sequence.
    log_k = pm.Normal("log_k", mu=0.0, sigma=1.0)
    k = pm.math.exp(log_k)

    # Shared guessing rate.
    lapse = pm.Beta("lapse", alpha=2.0, beta=18.0)

    c = suspicion[participant_id]
    post_a = pm.math.sigmoid(log_bf_a - c)
    post_b = pm.math.sigmoid(log_bf_b - c)
    post_ref = pm.math.sigmoid(BF_REF - c)
    slope_ref = post_ref * (1.0 - post_ref)
    p_engaged = pm.math.sigmoid(k * (post_a - post_b) / slope_ref)
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lapse + (1.0 - lapse) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
