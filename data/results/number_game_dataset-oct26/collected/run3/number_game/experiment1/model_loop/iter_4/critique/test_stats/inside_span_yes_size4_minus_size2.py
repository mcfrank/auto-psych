# name: inside_span_yes_size4_minus_size2
# description: Among non-example targets strictly inside the examples' span, yes-rate for 4-example sets minus for 2-example sets; observed above null means extra examples make people fill in the span more than the growing-evidence model predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    member = np.any(ex == t[:, None], axis=1)
    inside = (t > lo) & (t < hi) & ~member
    a, b = inside & (n == 4), inside & (n == 2)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
