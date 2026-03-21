import math


def balance_heuristic(stimulus, response_options):
    """
    Balance heuristic: sequences with more equal proportions of H and T are
    judged more random. Captures the folk belief that random sequences should
    have roughly equal outcomes.
    """
    seq_a, seq_b = stimulus

    def balance_score(seq):
        if len(seq) == 0:
            return 0.0
        p_h = seq.count("H") / len(seq)
        # 1.0 when perfectly balanced, 0.0 when all H or all T
        return 1.0 - abs(p_h - 0.5) * 2.0

    score_a = balance_score(seq_a)
    score_b = balance_score(seq_b)

    beta = 5.0
    exp_a = math.exp(beta * score_a)
    exp_b = math.exp(beta * score_b)
    total = exp_a + exp_b

    p_left = exp_a / total
    return {"left": p_left, "right": 1.0 - p_left}
