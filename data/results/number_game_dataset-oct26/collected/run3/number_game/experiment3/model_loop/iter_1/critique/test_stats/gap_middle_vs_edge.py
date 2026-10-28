# name: gap_middle_vs_edge
# description: For 2-example sets whose examples are at least 20 apart, among non-example targets strictly between them, yes-rate for targets within 5 of an example minus targets more than 5 from both; observed > null means people's gap-filling decays with distance from the examples (similarity) more than the model's flat interval membership predicts (observed < null: the middle is favoured).
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    E = pd.Series(uniq).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[codes]
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (n == 2) & (hi - lo >= 20) & (t > lo) & (t < hi)
    d = np.minimum(t - lo, hi - t)
    a = m & (d <= 5); b = m & (d > 5)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
