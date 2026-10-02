# name: streaky_pair_alternation_preference
# description: Among trials where both sequences have a longest run >= 4 and their alternation rates differ, the proportion choosing the MORE alternating sequence; observed above the null means that within streaky sequences people reward extra switches more than the model's quadratic ideal-rate term (flat near its far tail) predicts, below means less.
def test_statistic(df):
    def feats(s):
        best = cur = 1
        for x, y in zip(s, s[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1), best
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: feats(s) for s in u}
    aa = df["sequence_a"].map(lambda s: m[s][0]).to_numpy(); ab = df["sequence_b"].map(lambda s: m[s][0]).to_numpy()
    ra = df["sequence_a"].map(lambda s: m[s][1]).to_numpy(); rb = df["sequence_b"].map(lambda s: m[s][1]).to_numpy()
    sel = (ra >= 4) & (rb >= 4) & (np.abs(aa - ab) > 1e-9)
    if not sel.any():
        return 0.5
    cl = df["chose_left"].to_numpy()
    chose_more = np.where(aa > ab, cl, 1 - cl)
    return float(chose_more[sel].mean())
