"""
Alternation bias model (gambler's fallacy / negative recency).
People with this bias expect random sequences to alternate frequently,
so they judge sequences with more H-T and T-H transitions as "more random."
"""
import math


def alternation_bias(stimulus, response_options):
    """
    Prefers whichever sequence has the higher alternation rate (more H↔T transitions).
    Captures the systematic bias where people expect too much alternation in random sequences.
    """
    seq_a, seq_b = stimulus

    def alt_rate(seq):
        if len(seq) < 2:
            return 0.5
        return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]) / (len(seq) - 1)

    rate_a = alt_rate(seq_a)
    rate_b = alt_rate(seq_b)

    # Higher alternation rate → looks more random; logistic with scale 5
    diff = rate_a - rate_b
    p_left = 1.0 / (1.0 + math.exp(-diff * 5))

    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
