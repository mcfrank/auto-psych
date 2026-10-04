# name: switch_vs_balance_conflict
# description: Among pairs where one sequence has more switches but also a larger heads-tails count imbalance (cues conflict), the rate of choosing the higher-switch sequence; observed above null means people let switching override count balance more than the model predicts, below means count balance dominates more than the model allows.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    imb = {s: abs(2 * s.count("H") - len(s)) for s in seqs}
    ds = a.map(nsw) - b.map(nsw)
    di = a.map(imb) - b.map(imb)
    m = (ds * di) > 0
    if m.sum() == 0:
        return 0.5
    ch = np.where(ds[m] > 0, df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(ch))
