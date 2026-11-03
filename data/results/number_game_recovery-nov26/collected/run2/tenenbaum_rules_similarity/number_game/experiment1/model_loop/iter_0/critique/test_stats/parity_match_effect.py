# name: parity_match_effect
# description: Among non-member targets, yes-rate when target parity matches all examples minus when it does not; observed above null means humans weight parity similarity more than the model's odd/even rules produce, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()[:, None].astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    member = np.any(valid & (ex == t), axis=1)
    par = np.all(~valid | (np.mod(ex, 2) == np.mod(t, 2)), axis=1)
    a = (~member) & par; b = (~member) & ~par
    if not a.any() or not b.any(): return 0.0
    return float(r[a].mean() - r[b].mean())
