# name: range_inclusion_gap
# description: Among multi-example sets and targets not in the set, yes-rate for targets inside [min,max] of the examples minus yes-rate for targets outside; observed above model mean means people use numeric range/magnitude that the digit model ignores.
def test_statistic(df):
    X = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(X)
    n = valid.sum(axis=1)
    lo = np.nanmin(X, axis=1); hi = np.nanmax(X, axis=1)
    inset = (X == t[:, None]).any(axis=1)
    base = (n >= 2) & ~inset
    ins = base & (t >= lo) & (t <= hi); out = base & ~((t >= lo) & (t <= hi))
    if not ins.any() or not out.any():
        return 0.0
    return float(r[ins].mean() - r[out].mean())
