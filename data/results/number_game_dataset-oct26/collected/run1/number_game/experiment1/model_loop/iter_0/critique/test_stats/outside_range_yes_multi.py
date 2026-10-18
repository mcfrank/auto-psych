# name: outside_range_yes_multi
# description: Yes rate for targets outside [min, max] of the examples on sets with 3-4 examples; observed above null means the model is too confident that rich example sets exclude out-of-range numbers (under-generalizes beyond the range), below means it over-generalizes.
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
    m = (n >= 3) & ((t < lo) | (t > hi))
    return float(r[m].mean()) if m.any() else 0.0
