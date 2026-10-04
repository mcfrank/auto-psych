# name: terminal_effect_participant_sd
# description: SD across participants of each person's rate of choosing the sequence with the shorter final run, among pairs with equal switch counts and differing final runs; observed above null means people differ in recency weighting more than the model's shared terminal penalty allows (it under-produces heterogeneity), below means less.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    def last_run(s):
        k = 1
        while k < len(s) and s[-1 - k] == s[-1]:
            k += 1
        return k
    nsw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    lr = {s: last_run(s) for s in seqs}
    a = df["sequence_a"]; b = df["sequence_b"]
    d = (a.map(lr) - b.map(lr)).to_numpy()
    m = (a.map(nsw) == b.map(nsw)).to_numpy() & (d != 0)
    if m.sum() == 0:
        return 0.0
    y = df["chose_left"].to_numpy()[m]
    ch = np.where(d[m] < 0, y, 1 - y)
    r = pd.Series(ch).groupby(df["participant_id"].to_numpy()[m]).mean()
    return float(r.std(ddof=0))
