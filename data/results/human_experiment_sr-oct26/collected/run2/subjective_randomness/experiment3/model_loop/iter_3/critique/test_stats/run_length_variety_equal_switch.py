# name: run_length_variety_equal_switch
# description: Among pairs with equal numbers of switches whose sequences differ in the number of distinct run lengths, the rate of choosing the sequence with more distinct run lengths (mixed 1s, 2s, 3s); observed above null means people prize irregular run lengths more than the model predicts (it under-produces this choice), below means less.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def runs(s):
        out, c = [], 1
        for x, y in zip(s, s[1:]):
            if x == y:
                c += 1
            else:
                out.append(c); c = 1
        out.append(c)
        return out
    nsw = {s: len(runs(s)) for s in seqs}
    var = {s: len(set(runs(s))) for s in seqs}
    a = df["sequence_a"]; b = df["sequence_b"]
    dv = a.map(var) - b.map(var)
    m = (a.map(nsw) == b.map(nsw)) & (dv != 0)
    if m.sum() == 0:
        return 0.5
    ch = np.where(dv[m] > 0, df["chose_left"][m], 1 - df["chose_left"][m])
    return float(np.mean(ch))
