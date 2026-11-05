# name: outside_range_yes_rate
# description: Yes-rate for targets lying more than 10 beyond the [min, max] range of the examples; observed > null means the model under-generalizes far outside the example range (too interval-bound), observed < null means it over-generalizes there.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    lo, hi = np.nanmin(ex, axis=1), np.nanmax(ex, axis=1)
    m = (t < lo - 10) | (t > hi + 10)
    return float(df["rating"].to_numpy()[m].mean()) if m.any() else 0.0
