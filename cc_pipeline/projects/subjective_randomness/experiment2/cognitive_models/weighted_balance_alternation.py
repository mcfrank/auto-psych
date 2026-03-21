import math


def weighted_balance_alternation(stimulus, response_options):
    """
    Weighted combination of balance heuristic and alternation rate. Addresses
    balance_heuristic's core failure: it scores HTHTHTHT and HHTTHHTT identically
    because both are perfectly balanced, but participants strongly prefer the
    alternating sequence. This model adds an explicit alternation component.
    """
    seq_a, seq_b = stimulus

    def score(seq):
        if len(seq) == 0:
            return 0.0
        p_h = seq.count("H") / len(seq)
        balance = 1.0 - abs(p_h - 0.5) * 2.0
        if len(seq) < 2:
            alt = 0.5
        else:
            alt = sum(seq[i] != seq[i + 1] for i in range(len(seq) - 1)) / (len(seq) - 1)
        w = 0.5
        return w * balance + (1.0 - w) * alt

    score_a = score(seq_a)
    score_b = score(seq_b)

    beta = 5.0
    exp_a = math.exp(beta * score_a)
    exp_b = math.exp(beta * score_b)
    total = exp_a + exp_b

    p_left = exp_a / total
    return {"left": p_left, "right": 1.0 - p_left}
