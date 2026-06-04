import math


def alternation_bias(stimulus, response_options):
    """
    Monotonic alternation preference: sequences with higher alternation rates
    are judged more random. Captures the well-documented finding that people
    expect more alternations in random sequences than probability theory predicts.
    """
    seq_a, seq_b = stimulus

    def alternation_rate(seq):
        if len(seq) < 2:
            return 0.5
        return sum(seq[i] != seq[i + 1] for i in range(len(seq) - 1)) / (len(seq) - 1)

    rate_a = alternation_rate(seq_a)
    rate_b = alternation_rate(seq_b)

    beta = 5.0
    exp_a = math.exp(beta * rate_a)
    exp_b = math.exp(beta * rate_b)
    total = exp_a + exp_b

    p_left = exp_a / total
    return {"left": p_left, "right": 1.0 - p_left}
