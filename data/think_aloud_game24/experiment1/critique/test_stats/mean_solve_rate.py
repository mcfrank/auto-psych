def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int),
          and optionally lm_code_translation_list (list of str).
    Returns mean solve rate across all problems. Higher = more discrepancy if model under-predicts.
    """
    if not rows:
        return 0.0
    return sum(r["chose_left_pct"] for r in rows) / len(rows)
