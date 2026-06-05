def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int),
          and optionally lm_code_translation_list (list of str).
    Returns SD of solve rates across stimuli. Higher = more spread in difficulty distribution.
    A model that is too flat or too extreme will show a discrepancy here.
    """
    if len(rows) < 2:
        return 0.0
    rates = [r["chose_left_pct"] for r in rows]
    mean = sum(rates) / len(rates)
    variance = sum((x - mean) ** 2 for x in rates) / len(rates)
    return variance**0.5
