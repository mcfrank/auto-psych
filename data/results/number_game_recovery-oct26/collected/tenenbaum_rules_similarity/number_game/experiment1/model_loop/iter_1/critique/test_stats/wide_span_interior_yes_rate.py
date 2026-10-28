# name: wide_span_interior_yes_rate
# description: Yes rate for non-example targets strictly inside the examples range when the range is >=40 wide; observed below null_mean means the model over-fills wide spans with interval hypotheses (people treat sparse wide sets as rule-like, not as ranges).
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    mn = np.nanmin(s, axis=1); mx = np.nanmax(s, axis=1)
    n = np.sum(~np.isnan(s), axis=1)
    notin = ~np.any(s == t[:, None], axis=1)
    m = notin & (t > mn) & (t < mx) & ((mx - mn) >= 40) & (n >= 2)
    return float(r[m].mean()) if m.any() else 0.0
