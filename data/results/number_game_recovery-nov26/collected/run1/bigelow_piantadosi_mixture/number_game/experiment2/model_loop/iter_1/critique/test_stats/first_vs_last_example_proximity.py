# name: first_vs_last_example_proximity
# description: For sets of 2-4 examples, yes rate for non-example targets within 4 of the first-shown example (and not within 4 of the last) minus the same for the last-shown example; observed below null means the model's first-example (sticky guess) primacy is too strong / people weight later examples more.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(float).to_numpy()
    first = s.iloc[:, 0].to_numpy()
    last = s.ffill(axis=1).iloc[:, -1].to_numpy()
    is_ex = s.eq(pd.Series(t, index=s.index), axis=0).any(axis=1).to_numpy()
    nf = np.abs(t - first) <= 4
    nl = np.abs(t - last) <= 4
    base = (n >= 2) & (~is_ex)
    a = base & nf & ~nl
    b = base & nl & ~nf
    r = df["rating"].astype(float).to_numpy()
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
