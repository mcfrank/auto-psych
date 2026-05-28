def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int),
          and optionally lm_code_translation_list (list of str).
    Returns difference in mean solve rate between top and bottom quartile problems.
    Captures whether the model generates the right shape of difficulty distribution.
    """
    if len(rows) < 4:
        return 0.0
    rates = sorted(r["chose_left_pct"] for r in rows)
    n = len(rates)
    q1 = n // 4
    easy = rates[n - q1:] if q1 > 0 else rates[-1:]
    hard = rates[:q1] if q1 > 0 else rates[:1]
    if not easy or not hard:
        return 0.0
    return sum(easy) / len(easy) - sum(hard) / len(hard)
