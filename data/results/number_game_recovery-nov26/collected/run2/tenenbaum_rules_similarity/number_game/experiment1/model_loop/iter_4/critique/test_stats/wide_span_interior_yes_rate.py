# name: wide_span_interior_yes_rate
# description: For multi-example sets whose examples span at least 50 (max - min), yes-rate on targets strictly inside the span and not in the set; observed above null means people fill in wide ranges more than the Erlang preferred-width interval prior allows (it penalises wide intervals too much), below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    lo = np.nanmin(ex, axis=1)
    hi = np.nanmax(ex, axis=1)
    nex = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(float).to_numpy()
    inset = (ex == t[:, None]).any(axis=1)
    m = (nex >= 2) & (hi - lo >= 50) & (t > lo) & (t < hi) & ~inset
    r = df["rating"].to_numpy(dtype=float)
    return float(r[m].mean()) if m.sum() > 0 else 0.0
