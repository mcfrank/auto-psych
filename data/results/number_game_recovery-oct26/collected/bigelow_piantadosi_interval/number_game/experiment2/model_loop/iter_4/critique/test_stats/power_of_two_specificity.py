# name: power_of_two_specificity
# description: Among sets whose examples are all powers of 2 (2..64), yes rate to non-example powers of 2 minus yes rate to other even non-example targets; observed below the null means people generalise to the remaining powers of 2 less sharply (relative to evens) than the model, above means the model under-commits to the specific rule.
def test_statistic(df):
    p2 = {2, 4, 8, 16, 32, 64}
    sets = df["number_set"].astype(str)
    info = {}
    for s in sets.unique():
        e = [int(x) for x in s.split(",")]
        info[s] = set(e) <= p2
    all_p2 = sets.map(info).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    is_ex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)]) if all_p2.any() else np.zeros(len(t), bool)
    is_p2 = np.isin(t, list(p2))
    r = df["rating"].to_numpy().astype(float)
    a = all_p2 & ~is_ex & is_p2
    b = all_p2 & ~is_ex & ~is_p2 & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
