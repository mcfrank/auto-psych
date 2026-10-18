# name: inside_range_nonmember_yes_multi
# description: Yes rate for non-example targets lying strictly inside [min, max] of sets with 2+ examples; observed below null means the model over-fills the span between examples (interval-like generalization too strong), above means it under-fills it.
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4))
    ex = ex.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    n = (~np.isnan(ex)).sum(axis=1)
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    isin = (ex == t[:, None]).any(axis=1)
    m = (n >= 2) & (t > lo) & (t < hi) & ~isin
    return float(r[m].mean()) if m.any() else 0.0
