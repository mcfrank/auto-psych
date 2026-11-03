# name: power_set_inrange_nonpower_yes
# description: Among sets of 2+ examples that are all perfect powers (squares, cubes, powers of 2, incl. 1), yes rate for non-example targets that are not perfect powers and lie within the examples' [min,max]; observed above null means people fall back on a numeric range after power sets more than the model, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    pw = sorted(set([b ** e_ for b in range(1, 11) for e_ in range(2, 8) if b ** e_ <= 100] + [1]))
    isp = np.isin(ex, pw) | np.isnan(ex)
    lo = np.nanmin(ex, 1); hi = np.nanmax(ex, 1)
    sel = (k >= 2) & isp.all(1) & ~is_ex & ~np.isin(t, pw) & (t >= lo) & (t <= hi)
    return m(sel)
