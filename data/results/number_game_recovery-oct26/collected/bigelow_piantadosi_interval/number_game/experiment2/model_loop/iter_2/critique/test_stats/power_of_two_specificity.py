# name: power_of_two_specificity
# description: Among sets whose examples are all powers of 2 (2..64), yes rate to non-example powers of 2 minus yes rate to other even non-example targets; observed above the null means people commit to the specific "powers of 2" rule more sharply than the model (which leaks to "even"/ranges), below means the model over-commits to the specific rule.
def test_statistic(df):
    p2 = {"2", "4", "8", "16", "32", "64"}
    ex = df["number_set"].str.split(",")
    all_p2 = np.array([set(e) <= p2 for e in ex])
    t = df["target"].astype(int).to_numpy()
    tgt = t.astype(str)
    is_ex = np.array([s in e for s, e in zip(tgt, ex)])
    is_p2 = np.isin(t, [2, 4, 8, 16, 32, 64])
    r = df["rating"].to_numpy().astype(float)
    a = all_p2 & ~is_ex & is_p2
    b = all_p2 & ~is_ex & ~is_p2 & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
