def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int),
          and optionally lm_code_translation_list (list of str).
    Returns fraction of stimuli with solve rate < 0.3 (hard problems).
    Tests whether the model correctly predicts how many problems are genuinely difficult.
    Higher = more hard problems detected.
    """
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["chose_left_pct"] < 0.3) / len(rows)
