"""
Balance heuristic model.
People expect random sequences to contain roughly equal numbers of H and T,
so they judge sequences closer to 50/50 balance as "more random."
This is dissociable from alternation bias: a sequence like HHTTHHTT is
balanced but not highly alternating.
"""
import math


def balance_heuristic(stimulus, response_options):
    """
    Prefers whichever sequence has an H/T ratio closer to 0.5.
    Ignores the pattern of flips; only cares about overall frequency balance.
    """
    seq_a, seq_b = stimulus

    def balance_score(seq):
        if not seq:
            return 0.0
        ratio = seq.count("H") / len(seq)
        return -abs(ratio - 0.5)

    s_a = balance_score(seq_a)
    s_b = balance_score(seq_b)

    # diff in [-0.5, 0.5]; logistic with scale 8
    diff = s_a - s_b
    p_left = 1.0 / (1.0 + math.exp(-diff * 8))

    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
