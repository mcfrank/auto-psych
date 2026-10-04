# name: initial_streak_equal_switch
# description: Among pairs with equal numbers of switches and equal final-run length where the first runs differ in length, the rate of choosing the sequence with the shorter opening run; observed above null means people also penalise a sequence that starts on a streak (primacy) which the model, penalising only the terminal run, under-produces; below means the reverse.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def first_run(s):
        k = 1
        while k < len(s) and s[k] == s[0]:
            k += 1
        return k
    def last_run(s):
        return first_run(s[::-1])
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    fr = {s: first_run(s) for s in seqs}
    lr = {s: last_run(s) for s in seqs}
    a = df["sequence_a"]; b = df["sequence_b"]
    d = (a.map(fr) - b.map(fr)).to_numpy()
    m = ((a.map(nsw) == b.map(nsw)) & (a.map(lr) == b.map(lr))).to_numpy() & (d != 0)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where(d[m] < 0, y, 1 - y)))
