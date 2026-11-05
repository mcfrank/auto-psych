# name: inside_range_yes_rate
# description: Yes-rate for targets strictly inside the [min, max] span of the examples (excluding the examples themselves); observed > null means humans fill in the example span (interval-like generalization) more than the model predicts, observed < null means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    isex = np.any(ex == t[:, None], axis=1)
    m = (t > lo) & (t < hi) & ~isex
    return float(df["rating"].to_numpy()[m].mean()) if m.any() else 0.0
