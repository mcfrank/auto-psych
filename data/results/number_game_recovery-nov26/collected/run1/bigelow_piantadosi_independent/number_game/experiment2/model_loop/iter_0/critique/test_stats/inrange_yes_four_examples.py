# name: inrange_yes_four_examples
# description: Yes rate for non-example targets lying strictly between the min and max of a 4-example set; observed < null means the model over-fills the span between examples (too much interval generalisation) for large sets, observed > null means it under-fills.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    d = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    m = (n == 4) & (t > lo) & (t < hi) & (d > 0)
    return float(y[m].mean()) if m.sum() else 0.0
