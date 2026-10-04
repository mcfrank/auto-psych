# name: longest_run_shorter_choice_rate
# description: Among pairs whose sequences have the same number of switches but different longest-run lengths, the rate at which the sequence with the SHORTER longest run is chosen; observed above null means people penalise long streaks more than the model's span/switch terms imply.
def test_statistic(df):
    import itertools
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    lr = {s: max(len(list(g)) for _, g in itertools.groupby(s)) for s in seqs}
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) for s in seqs}
    la = df["sequence_a"].map(lr).to_numpy(); lb = df["sequence_b"].map(lr).to_numpy()
    sa = df["sequence_a"].map(sw).to_numpy(); sb = df["sequence_b"].map(sw).to_numpy()
    m = (sa == sb) & (la != lb)
    if m.sum() == 0:
        return 0.5
    c = df["chose_left"].to_numpy()[m]
    shorter_left = (la < lb)[m]
    return float(np.mean(np.where(shorter_left, c, 1 - c)))
