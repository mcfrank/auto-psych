# name: out_of_range_minus_in_range
# description: Yes rate for targets outside [min, max] of the examples minus yes rate for targets inside that range (sets with >=2 examples); observed above the null mean means the model over-weights interval/magnitude generalisation relative to humans (or under-generalises beyond the range).
def _parse(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    return ex, t, r

def test_statistic(df):
    ex, t, r = _parse(df)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    k = (~np.isnan(ex)).sum(axis=1) >= 2
    inr = k & (t >= lo) & (t <= hi); out = k & ~((t >= lo) & (t <= hi))
    if not inr.any() or not out.any():
        return 0.0
    return float(r[out].mean() - r[inr].mean())
