def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the standard deviation of chose_left_pct across stimuli.
    Higher = more spread in responses; tests whether model captures response variability.
    If the model is overconfident (predicts extreme probabilities), simulated SD will exceed
    observed SD, and this stat will not be significant. If model under-predicts dispersion,
    T_observed will exceed the null, flagging it as significant.
    """
    vals = [float(row["chose_left_pct"]) for row in rows]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return float(variance ** 0.5)
