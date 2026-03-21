def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the SD of chose_left_pct across stimuli — a measure of how much choice
    probabilities vary across stimuli.

    Models that predict a constant p_left (same for every stimulus) will produce
    simulated datasets with low stimulus-level variability (only binomial sampling
    noise). If the observed SD is much larger than what the model produces, that
    reveals the model cannot account for why some stimuli elicit strong preferences
    while others are near 50-50.

    Higher = more variability in choices across stimuli = more discrepancy for
    any model that predicts constant or near-constant preferences.
    """
    if not rows:
        return 0.0
    pcts = [float(r['chose_left_pct']) for r in rows]
    if len(pcts) < 2:
        return 0.0
    mean = sum(pcts) / len(pcts)
    variance = sum((p - mean) ** 2 for p in pcts) / len(pcts)
    return variance ** 0.5
