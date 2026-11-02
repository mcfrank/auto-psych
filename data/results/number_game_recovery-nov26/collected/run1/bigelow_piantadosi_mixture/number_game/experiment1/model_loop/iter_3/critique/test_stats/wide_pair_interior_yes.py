# name: wide_pair_interior_yes
# description: For two-example sets spanning >= 30, yes-rate for targets strictly between the two examples and more than 5 from both; observed < null means people fill in the gap between distant examples less than the model's interval hypotheses predict, observed > null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    n = np.isfinite(ex).sum(axis=1)
    t = df["target"].astype(float).to_numpy(); r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (n == 2) & (hi - lo >= 30) & (t > lo) & (t < hi) & (d > 5)
    if not m.any():
        return 0.0
    return float(r[m].mean())
