def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)

    Mean P(chose alternating sequence) across all stimuli.

    The alternating sequence in each pair is identified by its alternation rate == 1.0
    (every adjacent pair differs: HTHTHTHT or THTHTHTH). For each stimulus, compute the
    proportion of participants who chose the alternating sequence, then average across stimuli.

    Higher values = stronger preference for the alternating sequence.
    Catches models that underpredict alternating preference (griffiths variants, balance_heuristic).
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
            # sequence_a is the alternating one; chose_left = chose_alternating
            vals.append(p_left)
        elif ar_b > ar_a:
            # sequence_b is the alternating one; chose_right = chose_alternating
            vals.append(1.0 - p_left)
        else:
            # Tie: skip (shouldn't happen in this experiment)
            pass
    return sum(vals) / len(vals) if vals else 0.0
