# name: even_set_near_minus_far_even_target
# description: Among sets of 2+ all-even examples, yes rate for non-example even targets within 5 of some example minus even targets 15+ away from every example; observed above null means people show a numeric-similarity gradient inside the rule stronger than the model, below means weaker.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    e = np.nan_to_num(ex, nan=2.0)
    dist = np.nanmin(np.abs(ex - t[:, None]), 1)
    sel = (k >= 2) & (e % 2 == 0).all(1) & ~is_ex & (t % 2 == 0)
    return m(sel & (dist <= 5)) - m(sel & (dist >= 15))
