# name: wide_span_fill_in
# description: Yes-rate on in-span non-example targets for sets whose span (max-min) is >= 50 minus sets whose span is <= 25 (k >= 2); observed < null means people do not fill in wide spans the way the model's zero size mismatch inside the span predicts (model over-produces yes in wide spans), observed > null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), k - 1]
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    inspan = (t > mn) & (t < mx) & ~is_ex & (k >= 2)
    span = mx - mn
    return rate(inspan & (span >= 50)) - rate(inspan & (span <= 25))
