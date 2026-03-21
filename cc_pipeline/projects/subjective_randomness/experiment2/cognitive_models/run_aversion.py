import math


def run_aversion(stimulus, response_options):
    """
    Run aversion: sequences with shorter maximum runs are judged more random.
    Tests whether people specifically dislike long streaks (e.g. HHHHTTTT) rather
    than merely preferring alternation in general. Distinct from alternation_bias:
    a sequence like HTHTTTTH has moderate alternation but a long run of T, and
    these two models make different predictions for such cases.
    """
    seq_a, seq_b = stimulus

    def max_run_length(seq):
        if len(seq) == 0:
            return 0
        max_run = curr_run = 1
        for i in range(1, len(seq)):
            if seq[i] == seq[i - 1]:
                curr_run += 1
                if curr_run > max_run:
                    max_run = curr_run
            else:
                curr_run = 1
        return max_run

    def run_score(seq):
        if len(seq) <= 1:
            return 0.5
        # Normalized to [0, 1]: 1.0 when max_run=1 (fully alternating), 0.0 when one long run
        return 1.0 - (max_run_length(seq) - 1) / (len(seq) - 1)

    score_a = run_score(seq_a)
    score_b = run_score(seq_b)

    beta = 5.0
    exp_a = math.exp(beta * score_a)
    exp_b = math.exp(beta * score_b)
    total = exp_a + exp_b

    p_left = exp_a / total
    return {"left": p_left, "right": 1.0 - p_left}
