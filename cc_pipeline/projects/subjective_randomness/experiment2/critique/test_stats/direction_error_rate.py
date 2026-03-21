import math


def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns fraction of stimuli where run_aversion makes a confident direction error:
    model predicts P(left) > 0.60 but observed chose_left_pct < 0.50, OR
    model predicts P(left) < 0.40 but observed chose_left_pct > 0.50.
    Higher = more systematic reversals. When simulated data is generated from run_aversion
    itself, these errors should be rare (model rarely contradicts its own predictions),
    so significant p-values indicate genuine direction failures.
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

    errors = 0
    for row in rows:
        pred = model_pred(row["sequence_a"], row["sequence_b"])
        obs = float(row["chose_left_pct"])
        if (pred > 0.60 and obs < 0.50) or (pred < 0.40 and obs > 0.50):
            errors += 1

    return float(errors / len(rows)) if rows else 0.0
