# name: near_periodic_avoidance
# description: Among pairs with switch-rate difference at most 1/7 where exactly one sequence is near-periodic (period 2-4 with exactly one mismatch, not strictly periodic), rate of choosing the other sequence; observed above null_mean means people penalise almost-repeating patterns that the model's strict periodicity check misses.
def test_statistic(df):
    def alt(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y) / (len(s) - 1)
    def mism(s, p):
        return sum(1 for i in range(len(s) - p) if s[i] != s[i + p])
    def strict(s):
        n = len(s)
        return any(mism(s, p) == 0 for p in range(1, n // 2 + 1))
    def near(s):
        n = len(s)
        if n < 6 or strict(s):
            return 0
        return int(any(mism(s, p) == 1 for p in range(2, min(4, n // 2) + 1)))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    am = {s: alt(s) for s in u}
    nm = {s: near(s) for s in u}
    na, nb = df["sequence_a"].map(nm), df["sequence_b"].map(nm)
    d = (df["sequence_a"].map(am) - df["sequence_b"].map(am)).abs()
    m = (na != nb) & (d <= 1.0 / 7 + 1e-9)
    if m.sum() == 0:
        return 0.5
    chose_other = np.where(nb[m] == 1, df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(chose_other))
