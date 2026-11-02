# name: odd_mult3_set_nonmult3_vs_mult3
# description: Among sets of 2+ examples that are all odd multiples of 3 (e.g. 69,9,39,21; 63,27,99,33), yes rate for non-example odd targets that are NOT multiples of 3 minus non-example multiples of 3; observed above null means people read such sets as "odd" (or digit patterns) rather than "multiples of 3" more than the model, below means they favour "multiples of 3" more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = (~np.isnan(ex)).sum(1)
    is_ex = (ex == t[:, None]).any(1)
    P = set(n for n in range(2, 101) if all(n % d for d in range(2, int(n ** 0.5) + 1)))
    def m(mask):
        return float(y[mask].mean()) if mask.sum() > 0 else 0.0
    e = np.nan_to_num(ex, nan=3.0)
    sel = (k >= 2) & (e % 2 == 1).all(1) & (e % 3 == 0).all(1) & ~is_ex
    return m(sel & (t % 2 == 1) & (t % 3 != 0)) - m(sel & (t % 3 == 0))
