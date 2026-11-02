# name: mixed_parity_large_set_inspan_yes
# description: Yes-rate for non-example targets inside the examples' span in 3-4-example sets whose examples include both odd and even numbers; observed above null_mean means people fill in the range of heterogeneous sets more readily than the model (which keeps posterior mass on sparse rules).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    odd = np.where(valid, np.mod(ex, 2) == 1, False).sum(axis=1)
    mixed = (odd > 0) & (odd < n)
    t = df["target"].astype(float).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    inspan = (t >= np.nanmin(ex, axis=1)) & (t <= np.nanmax(ex, axis=1))
    m = (n >= 3) & mixed & inspan & ~inset
    r = df["rating"].astype(float).to_numpy()
    return float(r[m].mean()) if m.any() else 0.0
