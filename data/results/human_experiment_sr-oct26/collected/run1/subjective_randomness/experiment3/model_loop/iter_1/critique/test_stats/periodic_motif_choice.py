# name: periodic_motif_choice
# description: Among trials where exactly one sequence is a repeated block of period 3 or 4 (e.g. HHTHHT, HHTTHHTT, length >= twice the period, not perfectly alternating), the proportion choosing the periodic one; observed below null_mean means people detect and penalise repeating motifs beyond what the model's run/switch features predict.
def test_statistic(df):
    def periodic(s):
        n = len(s)
        if all(x != y for x, y in zip(s, s[1:])) or len(set(s)) < 2:
            return False
        for p in (3, 4):
            if n >= 2 * p and all(s[i] == s[i - p] for i in range(p, n)):
                return True
        return False
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    pf = {s: periodic(s) for s in seqs}
    pa = df["sequence_a"].map(pf).to_numpy(dtype=bool)
    pb = df["sequence_b"].map(pf).to_numpy(dtype=bool)
    y = df["chose_left"].to_numpy()
    m = pa != pb
    if m.sum() == 0:
        return 0.5
    return float(np.where(pa[m], y[m], 1 - y[m]).mean())
