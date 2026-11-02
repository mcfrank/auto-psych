# name: range_edge_asymmetry
# description: In 2-4 example sets, among non-example targets sharing no final digit with any example, yes-rate for targets 1-10 above the largest example minus targets 1-10 below the smallest example; observed above the null means people extend ranges upward more (relative to downward) than the model's compressed number line predicts, below means the model's upward bias is too strong or people favour smaller numbers.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    mn = np.nanmin(E, axis=1)
    mx = np.nanmax(E, axis=1)
    nondig = ~is_ex & ~np.any(Dg == np.mod(t, 10)[:, None], axis=1)
    a = (n >= 2) & nondig & (t > mx) & (t - mx <= 10)
    b = (n >= 2) & nondig & (t < mn) & (mn - t <= 10)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
