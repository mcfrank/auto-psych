# name: example_count_tightening
# description: Yes rate on non-example targets for 1-2-example sets minus for 3-4-example sets; observed above null means people tighten generalisation with more examples more than the tempered model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)  # (rows, <=4), NaN pad
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.abs(ex - t[:, None])
    nearest = np.nanmin(dist, axis=1)
    nonmember = nearest > 0
    n = np.sum(~np.isnan(ex), axis=1)
    a, b = r[nonmember & (n <= 2)], r[nonmember & (n >= 3)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
