# name: wide_span_inside_hull_yes_rate
# description: Among 3-4 example sets spanning at least 50 (max-min), yes rate for non-example targets lying inside [min,max]; observed above null means people generalize across the spanned range more than the model's interval/rule mixture predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    sel = (n >= 3) & (hi - lo >= 50) & (t >= lo) & (t <= hi) & ~np.any(ex == t[:, None], axis=1)
    return float(r[sel].mean()) if sel.any() else 0.0
