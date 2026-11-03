# name: single_example_yes_gap
# description: Yes-rate on non-member targets for one-example sets minus that for sets of 2-4 examples; observed above null means humans generalise from a single example more broadly (relative to larger sets) than the size principle predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    n = valid.sum(axis=1)
    a = (~member) & (n == 1); b = (~member) & (n >= 2)
    if not a.any() or not b.any(): return 0.0
    return float(r[a].mean() - r[b].mean())
