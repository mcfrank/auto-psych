def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)

    Mean P(chose comparison/non-alternating sequence) across all stimuli.
    Equals 1 - mean_chose_alternating.

    The comparison sequence is identified as the one that is NOT perfectly alternating
    (alt_rate < 1.0). For each stimulus, compute P(chose comparison sequence).

    Higher values = stronger preference for the comparison (non-alternating) sequence.
    Catches models that overpredict alternating preference (alternation_bias), which
    therefore underpredict comparison preference — generating very low T under the null,
    while the observed T is substantially higher.
    """
    def alt_rate(seq):
        if len(seq) < 2:
            return 0.0
        return sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1]) / (len(seq) - 1)

    vals = []
    for r in rows:
        p_left = float(r["chose_left_pct"])
        ar_a = alt_rate(r["sequence_a"])
        ar_b = alt_rate(r["sequence_b"])
        if ar_a > ar_b:
            # sequence_a is the alternating one; chose_right = chose_comparison
            vals.append(1.0 - p_left)
        elif ar_b > ar_a:
            # sequence_b is the alternating one; chose_left = chose_comparison
            vals.append(p_left)
        else:
            pass
    return sum(vals) / len(vals) if vals else 0.0
