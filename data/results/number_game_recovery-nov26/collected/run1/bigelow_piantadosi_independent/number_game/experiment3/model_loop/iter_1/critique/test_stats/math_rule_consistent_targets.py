# name: math_rule_consistent_targets
# description: In 2-4 example sets, yes-rate for non-example targets that share no final digit with any example but belong to a mathematical rule (squares, cubes, powers of 2-10, multiples of 3-12) containing every example; observed above the null means people endorse such rule-consistent numbers more than the model, below means the model over-weights these mathematical rules.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    n = np.sum(~np.isnan(E), axis=1)
    Dg = np.where(np.isnan(E), np.nan, np.mod(E, 10))
    is_ex = np.any(E == t[:, None], axis=1)
    nums = np.arange(1, 101)
    rules = [[k * k for k in range(1, 11)], [k ** 3 for k in range(1, 5)]]
    rules += [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    M = np.array([np.isin(nums, c) for c in rules])  # (R, 100)
    Ei = np.where(np.isnan(E), 1, E).astype(int) - 1
    memb = M[:, Ei]  # (R, rows, 4)
    memb = memb | np.isnan(E)[None, :, :]
    consistent = memb.all(axis=2)  # (R, rows)
    ti = np.clip(t.astype(int), 1, 100) - 1
    t_in = M[:, ti]  # (R, rows)
    hit = np.any(consistent & t_in, axis=0)
    nondig = ~is_ex & ~np.any(Dg == np.mod(t, 10)[:, None], axis=1)
    a = (n >= 2) & nondig & hit
    if a.sum() == 0:
        return 0.0
    return float(r[a].mean())
