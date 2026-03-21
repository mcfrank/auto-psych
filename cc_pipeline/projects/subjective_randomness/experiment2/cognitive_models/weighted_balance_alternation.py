"""
Weighted balance-alternation model.

Experiment 1 showed balance is the dominant cue (r = 0.63) while alternation alone
had no predictive power (r = -0.056). However, among balanced sequences, there was
some evidence that alternation influenced choices (e.g. THTHTH preferred over
moderately-balanced alternatives). This model combines both cues additively with
balance as the primary driver.
"""
import math


def weighted_balance_alternation(stimulus, response_options):
    """
    Scores sequences by a weighted sum of balance closeness to 0.5 (weight 0.7)
    and alternation rate closeness to 0.5 (weight 0.3).
    Balance is the primary cue; alternation breaks ties among balanced sequences.
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
        bal_dev = abs(balance_ratio(seq) - 0.5)
        alt_dev = abs(alt_rate(seq) - 0.5)
        # Lower deviation = higher score
        return -0.7 * bal_dev - 0.3 * alt_dev

    s_a = score(seq_a)
    s_b = score(seq_b)

    diff = s_a - s_b
    p_left = 1.0 / (1.0 + math.exp(-diff * 10))

    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
