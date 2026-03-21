def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns mean rate of choosing the more-alternating sequence (by consecutive-pair
    alternation fraction). Higher = stronger preference for alternating sequences.
    """

    def alt_rate(seq):
        """Fraction of consecutive pairs that alternate (differ)."""
        if len(seq) < 2:
            return 0.0
        alternations = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
        return alternations / (len(seq) - 1)

    pref_vals = []
    for row in rows:
        a_alt = alt_rate(row["sequence_a"])
        b_alt = alt_rate(row["sequence_b"])
        obs = float(row["chose_left_pct"])

        if a_alt > b_alt:
            # sequence_a is more alternating; chose_left_pct = rate of choosing it
            pref_vals.append(obs)
        elif b_alt > a_alt:
            # sequence_b is more alternating; (1 - chose_left_pct) = rate of choosing it
            pref_vals.append(1.0 - obs)
        # If equal alternation rates, skip this stimulus

    return float(sum(pref_vals) / len(pref_vals)) if pref_vals else 0.5
