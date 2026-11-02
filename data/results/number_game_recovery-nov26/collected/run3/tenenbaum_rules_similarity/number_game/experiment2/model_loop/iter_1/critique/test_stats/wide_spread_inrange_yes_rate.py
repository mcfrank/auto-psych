# name: wide_spread_inrange_yes_rate
# description: For sets of >=3 examples spanning >=60 (max-min), yes rate on targets strictly inside [min,max] that are not examples; observed > null means people still generalise to "almost any number in the span" more than the beta-share width prior (which penalises near-full ranges) predicts, observed < null means they generalise even less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    lo, hi = np.nanmin(E, axis=1), np.nanmax(E, axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    m = (n >= 3) & (hi - lo >= 60) & (t > lo) & (t < hi) & ~in_set
    return float(r[m].mean()) if m.sum() else 0.0
