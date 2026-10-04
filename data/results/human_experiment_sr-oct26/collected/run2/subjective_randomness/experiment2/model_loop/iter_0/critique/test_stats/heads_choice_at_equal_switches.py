# name: heads_choice_at_equal_switches
# description: Among pairs with equal switch counts and equal |#H-#T| imbalance but different head counts (mirror-type pairs), the rate of choosing the more-heads sequence; observed differing from null means the model's linear heads-share weight misstates the pure label asymmetry (above: model under-produces heads preference; below: over-produces).
def test_statistic(df):
    a, b = df["sequence_a"], df["sequence_b"]
    def sw(s):
        return s.map({x: sum(1 for p, q in zip(x, x[1:]) if p != q) for x in s.unique()})
    ha, hb = a.str.count("H"), b.str.count("H")
    ia = (2 * ha - a.str.len()).abs()
    ib = (2 * hb - b.str.len()).abs()
    mask = (sw(a) == sw(b)) & (ia == ib) & (ha != hb)
    if mask.sum() == 0:
        mask = (sw(a) == sw(b)) & (ha != hb)
    if mask.sum() == 0:
        return 0.5
    c = df["chose_left"][mask]
    return float(np.where((ha > hb)[mask], c, 1 - c).mean())
