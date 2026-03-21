import math


def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns mean absolute error between run_aversion model predictions and chose_left_pct.
    Higher = more discrepancy between model and data.
    """

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
        return 1.0 - (max_run_length(seq) - 1) / (len(seq) - 1)

    def model_pred(seq_a, seq_b, beta=5.0):
        sa = run_score(seq_a)
        sb = run_score(seq_b)
        ea = math.exp(beta * sa)
        eb = math.exp(beta * sb)
        return ea / (ea + eb)

    errors = []
    for row in rows:
        pred = model_pred(row["sequence_a"], row["sequence_b"])
        obs = float(row["chose_left_pct"])
        errors.append(abs(pred - obs))

    return float(sum(errors) / len(errors)) if errors else 0.0
