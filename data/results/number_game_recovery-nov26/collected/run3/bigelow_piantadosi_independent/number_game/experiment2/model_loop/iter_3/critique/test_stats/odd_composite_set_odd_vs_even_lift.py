# name: odd_composite_set_odd_vs_even_lift
# description: Among sets of 2+ examples that are all odd and include at least one odd composite (not all prime), yes rate for non-example odd targets minus non-example even targets; observed above null means people extend "odd numbers" more than the model (prime-over-odd penalty too strong), below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    e = np.nan_to_num(ex, nan=1.0)
    comp = np.isin(ex, [n for n in range(4, 101) if n not in P])
    sel = (k >= 2) & (e % 2 == 1).all(1) & comp.any(1) & ~is_ex
    return m(sel & (t % 2 == 1)) - m(sel & (t % 2 == 0))
