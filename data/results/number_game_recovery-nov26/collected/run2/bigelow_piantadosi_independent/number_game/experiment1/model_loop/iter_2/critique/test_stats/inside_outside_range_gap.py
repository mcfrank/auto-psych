# name: inside_outside_range_gap
# description: In sets with >=2 examples, yes rate for non-example targets inside [min,max] of the examples minus yes rate for targets outside that range; t_observed above null_mean means people fill in the span between examples (interval generalization) more sharply than the model, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    lo = np.nanmin(s, axis=1); hi = np.nanmax(s, axis=1)
    is_ex = (np.where(v, s, -1) == t[:, None]).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    base = (n >= 2) & ~is_ex
    ins = base & (t >= lo) & (t <= hi); out = base & ((t < lo) | (t > hi))
    if ins.sum() == 0 or out.sum() == 0:
        return 0.0
    return float(r[ins].mean() - r[out].mean())
