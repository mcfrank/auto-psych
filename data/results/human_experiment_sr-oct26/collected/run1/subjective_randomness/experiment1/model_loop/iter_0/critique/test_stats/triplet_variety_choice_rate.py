# name: triplet_variety_choice_rate
# description: Among trials where the sequences differ in the number of distinct length-3 substrings, the fraction choosing the sequence with more distinct triplets (pattern variety / complexity); observed > null means people value subpattern diversity beyond what balance and alternation capture.
def test_statistic(df):
    def variety(s):
        if len(s) < 3:
            return 0
        return len({s[i:i + 3] for i in range(len(s) - 2)})
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    m = {s: variety(s) for s in u}
    va = df["sequence_a"].map(m).to_numpy()
    vb = df["sequence_b"].map(m).to_numpy()
    y = df["chose_left"].to_numpy()
    mask = va != vb
    if mask.sum() == 0:
        return 0.5
    chose_more = np.where(va > vb, y, 1 - y)
    return float(chose_more[mask].mean())
