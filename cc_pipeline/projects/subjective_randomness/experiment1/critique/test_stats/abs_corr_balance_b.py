def test_stat(rows):
    """
    rows: list of dicts with keys: sequence_a, sequence_b, chose_left_pct (float), n (int)
    Returns the absolute Pearson correlation between chose_left_pct and the H/T balance
    of sequence_b (proportion of H in sequence_b).

    This tests whether participants' preferences are systematically explained by the
    balance (H/T ratio) of the comparison (right) sequence, independent of what any
    model predicts. If participants prefer the left sequence more often when sequence_b
    has an extreme H/T ratio, that suggests balance of the comparison matters — a cue
    that constant-prediction models ignore.

    Under any model with constant predictions, simulated choices are just binomial
    noise uncorrelated with sequence_b features → low correlation expected.
    If observed correlation is high, the model is missing a relevant feature.

    Higher = stronger absolute correlation = more discrepancy from models that
    ignore sequence_b balance.
    """
    if len(rows) < 3:
        return 0.0

    def balance(seq):
        if not seq:
            return 0.5
        return seq.count('H') / len(seq)

    pcts = [float(r['chose_left_pct']) for r in rows]
    bals = [balance(r['sequence_b']) for r in rows]

    n = len(pcts)
    mean_p = sum(pcts) / n
    mean_b = sum(bals) / n
    var_p = sum((p - mean_p) ** 2 for p in pcts)
    var_b = sum((b - mean_b) ** 2 for b in bals)

    if var_p == 0 or var_b == 0:
        return 0.0

    cov = sum((pcts[i] - mean_p) * (bals[i] - mean_b) for i in range(n))
    r = cov / (var_p * var_b) ** 0.5
    return abs(r)
