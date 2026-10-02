# name: perfect_alternation_choice
# description: Among trials where exactly one sequence is perfectly alternating (every flip switches, length >= 4), the proportion choosing the perfectly alternating one; observed below null_mean means people reject strict alternation more than the model's (leniently weighted, quadratic) over-alternation penalty predicts, above means they accept it more.
def test_statistic(df):
    def perfect(s):
        return len(s) >= 4 and all(x != y for x, y in zip(s, s[1:]))
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    pf = {s: perfect(s) for s in seqs}
    pa = df["sequence_a"].map(pf).to_numpy(dtype=bool)
    pb = df["sequence_b"].map(pf).to_numpy(dtype=bool)
    y = df["chose_left"].to_numpy()
    m = pa != pb
    if m.sum() == 0:
        return 0.5
    return float(np.where(pa[m], y[m], 1 - y[m]).mean())
