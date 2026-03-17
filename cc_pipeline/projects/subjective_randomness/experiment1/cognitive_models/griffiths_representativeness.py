"""
Griffiths-style rational representativeness model for subjective randomness.

Based on Griffiths & Tenenbaum's 'Randomness and coincidences: reconciling
intuition and probability theory'. Participants use a subjective alternating
generator (Markov chain with alternation probability q > 0.5) rather than a
fair coin as their mental model of randomness. A sequence looks 'more random'
if it has higher likelihood under this subjective generator.
"""

import math


def griffiths_representativeness(stimulus, response_options):
    """
    Prefer the sequence with higher likelihood under a subjective alternating
    Markov chain (p_alternation=0.7). Captures the rational basis for why
    people judge alternating sequences as 'more random'.
    """
    seq_a, seq_b = stimulus
    p_alt = 0.7  # subjective alternation probability

    def markov_likelihood(seq):
        if len(seq) == 0:
            return 1.0
        # First flip: uniform (0.5)
        log_p = -math.log(2)
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                log_p += math.log(p_alt)
            else:
                log_p += math.log(1.0 - p_alt)
        return math.exp(log_p)

    la = markov_likelihood(seq_a)
    lb = markov_likelihood(seq_b)
    total = la + lb
    if total <= 0:
        total = 1.0
    p_a = la / total
    return {response_options[0]: p_a, response_options[1]: 1.0 - p_a}
