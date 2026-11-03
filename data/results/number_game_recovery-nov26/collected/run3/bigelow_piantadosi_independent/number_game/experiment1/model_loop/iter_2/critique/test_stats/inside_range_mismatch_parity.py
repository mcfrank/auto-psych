# name: inside_range_mismatch_parity
# description: Among 2+ example sets, yes rate for targets strictly inside the examples' [min,max] span that share NO common structure with the examples (parity differs from a parity-uniform set or set is mixed-parity) minus the same for outside-span targets; observed > null means people use a magnitude/interval region beyond what rule-based concepts allow and the model under-produces it.
def test_statistic(df):
    s = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ex = s.str.split(",", expand=True)
    n_ex = (s.str.count(",") + 1).values
    vals = ex.apply(pd.to_numeric, errors="coerce").values.astype(float)
    lo = np.nanmin(vals, axis=1); hi = np.nanmax(vals, axis=1)
    par = np.where(np.isnan(vals), np.nan, vals % 2)
    pmin = np.nanmin(par, axis=1); pmax = np.nanmax(par, axis=1)
    uniform = pmin == pmax
    mismatch = (~uniform) | (t % 2 != pmin)
    is_ex = np.any(vals == t[:, None], axis=1)
    base = (n_ex >= 2) & mismatch & ~is_ex
    a = base & (t > lo) & (t < hi)
    b = base & ((t < lo) | (t > hi))
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
