def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the standard deviation of chose_left_pct across stimuli.
    Higher = more spread in responses; tests whether model captures response variability.
    """
    vals = [float(row["chose_left_pct"]) for row in rows]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    return float(variance ** 0.5)
