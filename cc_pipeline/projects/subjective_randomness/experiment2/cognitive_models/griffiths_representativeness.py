"""
Griffiths-style rational representativeness model.
A sequence looks "more random" when its statistics (alternation rate and H/T balance)
are closer to those expected from a fair coin (both should be ~0.5).
See: Griffiths & Tenenbaum, "Randomness and Coincidences" (2001).
"""
import math


def _alternation_rate(seq):
    if len(seq) < 2:
        return 0.5
    return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]) / (len(seq) - 1)


def _balance_ratio(seq):
    if not seq:
        return 0.5
    return seq.count("H") / len(seq)


def griffiths_representativeness(stimulus, response_options):
    """
    Prefers the sequence whose alternation rate and H/T balance are both closer
    to 0.5 — the statistics expected from a fair-coin generator. Combines both
    dimensions equally (squared deviation from 0.5).
    """
    seq_a, seq_b = stimulus

    def score(seq):
        alt = _alternation_rate(seq)
        bal = _balance_ratio(seq)
        return -(alt - 0.5) ** 2 - (bal - 0.5) ** 2

    s_a = score(seq_a)
    s_b = score(seq_b)

    # Logistic with scale factor 8; diff in [~-0.5, ~0.5]
    diff = s_a - s_b
    p_left = 1.0 / (1.0 + math.exp(-diff * 8))

    return {response_options[0]: p_left, response_options[1]: 1.0 - p_left}
