# name: late_switch_concentration_choice
# description: Among trials whose sequences differ in (switches in the second half minus switches in the first half), the proportion choosing the sequence whose switching is concentrated LATER (streaky start, alternating end); observed above null_mean means people judge sequences by how they end/begin in a way the order-blind switch-rate term misses (below: they prefer early alternation).
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    def late(s):
        u = pd.unique(s)
        def f(x):
            t = [int(x[i] != x[i + 1]) for i in range(len(x) - 1)]
            h = len(t) // 2
            return sum(t[len(t) - h:]) - sum(t[:h])
        return s.map({x: f(x) for x in u}).to_numpy()
    la, lb = late(a), late(b)
    mask = la != lb
    if not mask.any():
        return 0.5
    c = df["chose_left"].to_numpy()
    chose_late = np.where(la > lb, c, 1 - c)
    return float(chose_late[mask].mean())
