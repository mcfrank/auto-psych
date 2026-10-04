# name: more_distinct_triplets_at_equal_switches
# description: Among pairs with equal switch counts but different numbers of distinct length-3 substrings, the rate of choosing the sequence with more distinct triplets (richer local patterns); observed above null means people reward local pattern variety that the alternation/periodicity/balance terms miss.
def test_statistic(df):
    a = df["sequence_a"].to_numpy(); b = df["sequence_b"].to_numpy()
    u = pd.unique(np.concatenate([a, b]))
    tri = {s: len({s[i:i + 3] for i in range(max(len(s) - 2, 0))}) for s in u}
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in u}
    ta = df["sequence_a"].map(tri).to_numpy(); tb = df["sequence_b"].map(tri).to_numpy()
    swa = df["sequence_a"].map(sw).to_numpy(); swb = df["sequence_b"].map(sw).to_numpy()
    m = (swa == swb) & (ta != tb)
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.where(ta[m] > tb[m], c, 1 - c).mean())
