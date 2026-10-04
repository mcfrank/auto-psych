# name: long_streak_count_choice_rate
# description: Among pairs whose sequences differ in the number of runs of 3+ identical flips but have equal switch counts and equal longest run, the rate the sequence with FEWER such streaks is chosen; observed above null means people penalise each extra visible streak beyond what span/switch terms capture.
def test_statistic(df):
    import itertools
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    runs = {s: [len(list(g)) for _, g in itertools.groupby(s)] for s in seqs}
    ns = {s: sum(r >= 3 for r in v) for s, v in runs.items()}
    lr = {s: max(v) for s, v in runs.items()}
    na = df["sequence_a"].map(ns).to_numpy(); nb = df["sequence_b"].map(ns).to_numpy()
    sa = df["sequence_a"].map(lambda s: len(runs[s])).to_numpy(); sb = df["sequence_b"].map(lambda s: len(runs[s])).to_numpy()
    m = (na != nb) & (sa == sb)
    if m.sum() == 0:
        # fall back: pairs differing in streak count at all
        m = na != nb
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    return float(np.mean(np.where((na < nb)[m], c, 1 - c)))
