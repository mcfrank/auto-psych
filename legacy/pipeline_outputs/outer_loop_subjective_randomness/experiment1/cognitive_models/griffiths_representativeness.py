import math


def griffiths_representativeness(stimulus, response_options):
    """
    Griffiths & Tenenbaum representativeness model: a sequence is judged random
    to the extent that both its alternation rate and its H/T balance are close
    to the expected values for a fair coin (0.5 each). Sequences that deviate
    from either expected statistic are judged less random.
    """
    seq_a, seq_b = stimulus

    def representativeness(seq):
        if len(seq) == 0:
            return 0.0
        # Balance deviation
        p_h = seq.count("H") / len(seq)
        balance_dev = abs(p_h - 0.5)
        # Alternation deviation
        if len(seq) < 2:
            alt_dev = 0.0
        else:
            alt_rate = sum(seq[i] != seq[i + 1] for i in range(len(seq) - 1)) / (len(seq) - 1)
            alt_dev = abs(alt_rate - 0.5)
        # Higher representativeness when both statistics are close to 0.5
        return -(balance_dev + alt_dev)

    rep_a = representativeness(seq_a)
    rep_b = representativeness(seq_b)

    beta = 5.0
    exp_a = math.exp(beta * rep_a)
    exp_b = math.exp(beta * rep_b)
    total = exp_a + exp_b

    p_left = exp_a / total
    return {"left": p_left, "right": 1.0 - p_left}
