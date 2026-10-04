# name: run_length_variety_equal_switch_terminal
# description: Among pairs with equal numbers of switches and equal final-run length whose sequences differ in the number of distinct run lengths, the rate of choosing the sequence with more distinct run lengths; observed above null means people prize irregular run lengths more than the model (which has no run-length-variety cue beyond triplets) predicts, below means less.
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
    R = {s: runs(s) for s in seqs}
    nsw = {s: len(r) for s, r in R.items()}
    var = {s: len(set(r)) for s, r in R.items()}
    last = {s: r[-1] for s, r in R.items()}
    a = df["sequence_a"]; b = df["sequence_b"]
    dv = a.map(var) - b.map(var)
    m = (a.map(nsw) == b.map(nsw)) & (a.map(last) == b.map(last)) & (dv != 0)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m.to_numpy()]
    ch = np.where(dv[m].to_numpy() > 0, y, 1 - y)
    return float(np.mean(ch))
