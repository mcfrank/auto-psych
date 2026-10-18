# name: square_target_boost_in_square_sets
# description: In sets where at least two examples are perfect squares (and >=half are), yes-rate on non-example perfect-square targets minus on non-square targets; observed above model mean means people infer a 'squares' rule the digit/common-factor model lacks.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    sq = lambda a: np.isclose(np.round(np.sqrt(np.nan_to_num(a))) ** 2, np.nan_to_num(a))
    nsq = (sq(X) & valid).sum(axis=1)
    sset = (nsq >= 2) & (nsq * 2 >= n)
    inset = (X == t[:, None]).any(axis=1)
    ts = sq(t)
    a = sset & ~inset & ts; b = sset & ~inset & ~ts
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
