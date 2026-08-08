"""Alternation-surprise accumulator.

Cognitive hypothesis (one mechanism):
People judge how random a sequence is by reading it toss by toss and
accumulating the surprise of each new outcome given the outcome before it.
Because they hold a mild expectation that outcomes will alternate rather
than repeat (a weak gambler's-fallacy bias), a repeat feels more surprising
the longer the current run of identical outcomes already is. A sequence
that accumulates less total surprise — one that matched that alternation
expectation more closely — is judged more random and preferred.

Signature: p(chose_left) rises with (surprise_b - surprise_a): the sequence
with lower accumulated alternation-surprise is the more random one.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _accumulated_surprise(seq: str) -> float:
    """Total alternation-biased surprisal of a H/T sequence, read left-to-right.

    Belief about the probability that the next toss repeats the current one,
    given the length r of the run already in place (r>=1 at the first toss):
        p_same(r) = 1 / (1 + exp(0.6 * r)),
    so repetition becomes increasingly surprising as a run grows.

    Surprise per toss (t >= 2) is -log(P(observed outcome)) in nats under that
    belief; repeated outcomes draw on p_same(r), alternating outcomes on
    1 - p_same(r). The scalar returned is the sum over the run.
    """
    toks = [c.strip().upper() for c in str(seq).strip()]
    if not toks:
        return 0.0
    n = len(toks)
    if n < 2:
        return 0.0

    surprise = 0.0
    # run length of consecutive identical outcomes ending before the current toss
    run = 1
    for i in range(1, n):
        p_same = 1.0 / (1.0 + np.exp(0.6 * run))
        p_same_l = min(max(p_same, 1e-6), 1.0 - 1e-6)  # keep finite
        if toks[i] == toks[i - 1]:
            surprise += -np.log(p_same_l)
            run += 1
        else:
            surprise += -np.log(1.0 - p_same_l)
            run = 1
    return float(surprise)


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Expose per-sequence accumulated alternation-surprise for one stimulus pair."""
    return {
        "surprise_a": _accumulated_surprise(sequence_a),
        "surprise_b": _accumulated_surprise(sequence_b),
    }


with pm.Model() as model:
    # Stimulus inputs: accumulated alternation-surprise (from compute_features).
    surprise_a = pm.Data("surprise_a", np.zeros(1, dtype="float64"))
    surprise_b = pm.Data("surprise_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters.
    # tau > 0: how strongly accumulated-surprise differences drive the choice.
    # beta: a baseline response bias (which stimulus a participant starts from).
    tau = pm.HalfNormal("tau", sigma=1.5)
    beta = pm.Normal("beta", mu=0.0, sigma=0.5)

    # Lower accumulated alternation-surprise => judged more random => preferred.
    # Positive (surprise_b - surprise_a) means A is the more random sequence,
    # so it pushes the log-odds of choosing left upward.
    logit_p = beta + tau * (surprise_b - surprise_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit_p))

    # Observed response: pass the pm.Data tensor directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
