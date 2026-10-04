# name: balanced_choice_at_equal_switches
# description: Among pairs whose two sequences have the same number of switches but different |#H-#T| imbalance, the rate of choosing the more balanced sequence; observed above null means the model under-weights heads/tails balance (it only has a linear heads-share term, no symmetric balance term).
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    def sw(s):
        u = pd.Series(s.unique())
        m = {x: sum(1 for p, q in zip(x, x[1:]) if p != q) for x in u}
        return s.map(m)
    ia = (2 * a.str.count("H") - a.str.len()).abs()
    ib = (2 * b.str.count("H") - b.str.len()).abs()
    mask = (sw(a) == sw(b)) & (ia != ib)
    if mask.sum() == 0:
        return 0.5
    left_more_bal = (ia < ib)[mask]
    c = df["chose_left"][mask]
    return float(np.where(left_more_bal, c, 1 - c).mean())
