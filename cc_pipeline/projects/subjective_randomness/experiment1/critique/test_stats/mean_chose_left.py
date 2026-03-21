def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the mean proportion of participants who chose the LEFT sequence (sequence_a)
    across all stimuli. Computed as mean(chose_left_pct).

    This test detects models that systematically under-predict preference for the
    left sequence. If a model predicts p(left) ~ 0.19 (e.g., griffiths_representativeness,
    which predicts HTHT/THTH sequences are NOT random) but participants actually chose
    left 38%+ of the time, the observed mean_chose_left will be much higher than
    what the model generates in simulated data.

    Higher = participants chose left more often = more discrepancy for models that
    under-predict left-sequence preference.
    """
    if not rows:
        return 0.0
    return sum(float(r['chose_left_pct']) for r in rows) / len(rows)
