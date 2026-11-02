# name: last_vs_first_parity_match_gap
# description: Among sets whose first- and last-shown examples differ in parity, yes rate for non-example targets matching the LAST example's parity minus targets matching the FIRST example's parity; t_observed above null_mean means recency acts on rule (parity) evidence more strongly than the model's fitted order gradient produces, below means less (or primacy).
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    first = ex[:, 0]
    last = ex[np.arange(len(ex)), n - 1]
    isex = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & ((first % 2) != (last % 2)) & ~isex
    r = df["rating"].to_numpy(dtype=float)
    a = m & ((t % 2) == (last % 2))
    b = m & ((t % 2) == (first % 2))
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
