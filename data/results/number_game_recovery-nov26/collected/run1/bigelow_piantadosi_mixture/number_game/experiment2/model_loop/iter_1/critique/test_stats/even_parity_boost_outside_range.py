# name: even_parity_boost_outside_range
# description: For sets of >=2 examples that are all even, yes rate for even targets outside the examples' range minus odd targets outside it; observed above null means people generalise an arithmetic parity/multiples rule beyond the range more than the model predicts (below: the model over-weights such rules).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    ev = ((s % 2 == 0) | s.isna()).all(axis=1).to_numpy() & (n >= 2)
    mn = s.min(axis=1).to_numpy(); mx = s.max(axis=1).to_numpy()
    out = (t < mn) | (t > mx)
    r = df["rating"].astype(float).to_numpy()
    a = ev & out & (t % 2 == 0); b = ev & out & (t % 2 == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
