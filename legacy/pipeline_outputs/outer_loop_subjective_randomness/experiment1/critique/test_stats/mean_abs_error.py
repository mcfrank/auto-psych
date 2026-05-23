import math


def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns mean absolute error between balance_heuristic prediction and chose_left_pct.
    Higher = more discrepancy between model and data.
    """

    def balance_score(seq):
        if not seq:
            return 0.0
        p_h = seq.count("H") / len(seq)
        return 1.0 - abs(p_h - 0.5) * 2.0

    def model_pred(seq_a, seq_b, beta=5.0):
        sa = balance_score(seq_a)
        sb = balance_score(seq_b)
        ea = math.exp(beta * sa)
        eb = math.exp(beta * sb)
        return ea / (ea + eb)

    errors = []
    for row in rows:
        pred = model_pred(row["sequence_a"], row["sequence_b"])
        obs = float(row["chose_left_pct"])
        errors.append(abs(pred - obs))

    return float(sum(errors) / len(errors)) if errors else 0.0
