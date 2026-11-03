# name: wide_set_interior_gap_fill
# description: For sets of >=3 examples spanning >=50, yes rate for targets strictly inside the span but >=10 from every example; observed > null means people fill in the interior of widely spread examples more than the model predicts (under-weights broad ranges), observed < null less.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    absd = np.abs(E - t[:, None])
    d = np.nanmin(absd, axis=1)
    ne = E[np.arange(len(t)), np.nanargmin(absd, axis=1)]
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    m = (n >= 3) & ((hi - lo) >= 50) & (d >= 10) & (t > lo) & (t < hi)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
