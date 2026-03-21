def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the mean proportion of participants who chose the RIGHT sequence (sequence_b)
    across all stimuli. Computed as mean(1 - chose_left_pct).

    This test detects models that systematically over-predict the left sequence as
    'more random'. If a model predicts p(left) ~ 0.9 but participants actually chose
    right 60%+ of the time, the observed mean_chose_right will be much higher than
    what the model generates in simulated data.

    Higher = participants chose right more often = more discrepancy for models that
    over-predict left-sequence preference.
    """
    if not rows:
        return 0.0
    return sum(1.0 - float(r['chose_left_pct']) for r in rows) / len(rows)
