def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns mean rate of choosing the more-alternating sequence, measured by alternation
    fraction (consecutive-pair differences / total pairs). For each stimulus, if sequence_a
    alternates more, uses chose_left_pct; if sequence_b alternates more, uses 1 - chose_left_pct.
    Stimuli with equal alternation rates are excluded.
    Higher = stronger preference for alternating sequences.
    Tests whether run_aversion's max-run criterion tracks the alternation preference seen in data.
    """

    def alt_rate(seq):
        if len(seq) < 2:
            return 0.0
        return sum(seq[i] != seq[i + 1] for i in range(len(seq) - 1)) / (len(seq) - 1)

    pref_vals = []
    for row in rows:
        a_alt = alt_rate(row["sequence_a"])
        b_alt = alt_rate(row["sequence_b"])
        obs = float(row["chose_left_pct"])
        if a_alt > b_alt:
            pref_vals.append(obs)
        elif b_alt > a_alt:
            pref_vals.append(1.0 - obs)

    return float(sum(pref_vals) / len(pref_vals)) if pref_vals else 0.5
