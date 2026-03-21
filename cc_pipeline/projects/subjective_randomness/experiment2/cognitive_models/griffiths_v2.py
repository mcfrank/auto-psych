"""
Griffiths-style representativeness, variant 2: softer alternation prior (p_alt = 0.6).

Experiment 1 used p_alt = 0.7 (both alternation and balance penalized equally via
squared deviation from 0.5). That model showed near-zero correlation (r = -0.037),
possibly because p_alt = 0.7 over-weights alternation. This variant down-weights
alternation deviation by half, making balance the dominant cue while alternation
contributes a smaller secondary penalty — consistent with experiment 1 findings.
"""
import math


def griffiths_v2(stimulus, response_options):
    """
    Prefers the sequence whose statistics are closer to a fair coin, but with
    balance deviation penalized more heavily (weight 2x) than alternation deviation.
    Motivated by experiment 1: balance was the dominant predictor, alternation was not.
    """
    seq_a, seq_b = stimulus

    def alt_rate(seq):
        if len(seq) < 2:
            return 0.5
        return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]) / (len(seq) - 1)

    def balance_ratio(seq):
        if not seq:
            return 0.5
        return seq.count("H") / len(seq)

    def score(seq):
        alt = alt_rate(seq)
        bal = balance_ratio(seq)
        # Balance gets 2x weight; alternation gets 0.5x weight
        return -2.0 * (bal - 0.5) ** 2 - 0.5 * (alt - 0.5) ** 2

    s_a = score(seq_a)
    s_b = score(seq_b)

    diff = s_a - s_b
    p_left = 1.0 / (1.0 + math.exp(-diff * 8))

    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
