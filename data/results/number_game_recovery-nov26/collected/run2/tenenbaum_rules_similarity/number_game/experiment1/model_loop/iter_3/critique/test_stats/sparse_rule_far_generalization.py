# name: sparse_rule_far_generalization
# description: Among non-member targets more than 15 from every example, yes-rate for targets belonging to a sparse rule (multiples of 6-12, squares, cubes, powers of 2-10; size <= 17) that contains all of a set's >=2 examples, minus yes-rate for far targets outside every such rule; observed below null means people extend sparse arithmetic rules to remote numbers less than the model's rule posterior predicts, above null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    exi = np.where(valid, ex, 1).astype(int)
    member = np.nansum(ex == t[:, None], axis=1) > 0
    mind = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    rules = [set(range(m, 101, m)) for m in range(6, 13)]
    rules.append({k * k for k in range(1, 11)})
    rules.append({k ** 3 for k in range(1, 5)})
    for b in range(2, 11):
        rules.append({b ** k for k in range(1, 8) if b ** k <= 100})
    in_rule = np.zeros(len(t), dtype=bool)
    for R in rules:
        mem = np.zeros(101, dtype=bool); mem[list(R)] = True
        all_in = np.all(mem[exi] | ~valid, axis=1) & (n >= 2)
        in_rule |= all_in & mem[t]
    far = (~member) & (mind > 15)
    a = far & in_rule; b = far & ~in_rule
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
