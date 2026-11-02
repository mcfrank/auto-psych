# name: single_example_above_minus_below
# description: For one-example sets, yes rate for targets above the example minus yes rate for targets below it (target != example); observed above null means people generalise upward more than the model's symmetric range prior predicts, below means downward more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    s1 = (k == 1)
    a = s1 & (t > lo); b = s1 & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
