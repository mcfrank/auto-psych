# name: unfamiliar_multiple_rule_gap
# description: In sets of >=2 examples, not all even and not all multiples of 5, whose examples are all multiples of some m in {3,7,9,11} (largest such m), yes rate for non-example multiples of m minus non-multiples; observed below null means people generalize by unfamiliar multiples rules less than the model predicts, above means more.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    m_sel = np.zeros(len(t))
    for m in [3, 7, 9, 11]:
        ok = ((E % m == 0) | pad).all(axis=1)
        m_sel = np.where(ok, m, m_sel)
    all_even = ((E % 2 == 0) | pad).all(axis=1)
    all_five = ((E % 5 == 0) | pad).all(axis=1)
    base = (n >= 2) & (m_sel > 0) & ~all_even & ~all_five & ~in_set
    inrule = (t % np.where(m_sel > 0, m_sel, 1)) == 0
    return rate(base & inrule) - rate(base & ~inrule)
