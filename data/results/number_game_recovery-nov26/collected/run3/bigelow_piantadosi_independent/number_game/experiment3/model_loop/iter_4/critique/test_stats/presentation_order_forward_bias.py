# name: presentation_order_forward_bias
# description: On 2+ example sets, (yes above max minus yes below min) for sets shown in ascending order minus the same for sets shown in descending order; observed above null means the upward lean follows the direction the examples were shown in (which the order-blind model misses), below null means descending order strengthens it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").values.astype(float)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    n = (~np.isnan(ex)).sum(1)
    mx = np.nanmax(ex, 1); mn = np.nanmin(ex, 1)
    isex = (ex == t[:, None]).any(1)
    k = ex.shape[1]
    d = np.diff(ex, axis=1)
    valid = ~np.isnan(d)
    asc = (n >= 2) & np.where(valid, d > 0, True).all(1)
    desc = (n >= 2) & np.where(valid, d < 0, True).all(1)
    up = (t > mx) & ~isex; dn = (t < mn) & ~isex
    def gap(m):
        a = r[m & up]; b = r[m & dn]
        if len(a) == 0 or len(b) == 0:
            return 0.0
        return a.mean() - b.mean()
    return float(gap(asc) - gap(desc))
