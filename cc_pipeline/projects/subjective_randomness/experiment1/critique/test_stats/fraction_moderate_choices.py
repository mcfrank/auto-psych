def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the fraction of stimuli where chose_left_pct is 'moderate' (between 0.3 and 0.7
    inclusive), meaning neither strongly left nor strongly right.

    A model that predicts an extreme p(left) (e.g., 0.946 or 0.189) will rarely
    generate simulated datasets where any individual stimulus lands near 50-50.
    If the observed data has many stimuli with moderate, split choices, while the
    model's synthetic data is dominated by extreme choices, this reveals that the
    model is overconfident — it predicts a much stronger directional preference than
    participants actually show.

    Higher = more stimuli with moderate, near-even splits = more discrepancy for
    overconfident models.
    """
    if not rows:
        return 0.0
    count = sum(1 for r in rows if 0.3 <= float(r['chose_left_pct']) <= 0.7)
    return count / len(rows)
