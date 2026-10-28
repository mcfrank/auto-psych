# name: high_minus_low_target_bias
# description: Among non-example targets more than 10 from every example, yes rate for targets above 50 minus targets at or below 50; observed above null_mean means people favour large numbers (or the model under-endorses them) beyond what the magnitude-neutral concept model predicts, below means people favour small numbers.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    sel = dist > 10
    a, b = sel & (t > 50), sel & (t <= 50)
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
