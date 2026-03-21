def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)

    Fraction of stimuli where the choice was NOT unanimous (chose_left_pct strictly
    between 0 and 1). With n=5 per stimulus this means 1, 2, 3, or 4 participants
    chose left (chose_left_pct in {0.2, 0.4, 0.6, 0.8}).

    A value of 1.0 means all stimuli produced split decisions.
    A value of 0.0 means all stimuli produced unanimous decisions.

    Higher T = more stimuli with mixed preferences (near even splits).
    Catches overconfident models (alternation_bias and griffiths variants) that predict
    extreme P(left) ≈ 0.95 or ≈ 0.05, which almost never produces split outcomes.
    Balance_heuristic (p=0.5 everywhere) predicts HIGHER fraction moderate than observed,
    so it does NOT generate values below the observed T — not caught by this test.
    """
    if not rows:
        return 0.0
    moderate = sum(
        1 for r in rows
        if 0.0 < float(r["chose_left_pct"]) < 1.0
    )
    return moderate / len(rows)
