def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns mean chose_left_pct across all stimuli. Tests for systematic left/right bias.
    Higher = stronger leftward bias in the data.
    """
    vals = [float(row["chose_left_pct"]) for row in rows]
    return float(sum(vals) / len(vals)) if vals else 0.5
