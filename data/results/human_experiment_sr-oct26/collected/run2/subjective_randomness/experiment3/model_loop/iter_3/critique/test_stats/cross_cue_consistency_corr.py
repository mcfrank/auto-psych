# name: cross_cue_consistency_corr
# description: Across participants, Pearson correlation between a person's rate of agreeing with the pair majority on pairs that differ in switch rate (|diff| >= 0.25) and on pairs with equal switch rate; observed above null means a single person-level consistency/noise trait spans all cues, which the model (independent per-cue person weights) under-produces.
def test_statistic(df):
    seqs = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1) for s in seqs}
    a = df["sequence_a"]; b = df["sequence_b"]
    first = a < b
    key = pd.Series(np.where(first, a + "|" + b, b + "|" + a), index=df.index)
    y = pd.Series(np.where(first, df["chose_left"], 1 - df["chose_left"]), index=df.index)
    rate = y.groupby(key).transform("mean")
    agree = ((y == 1) == (rate >= 0.5)).astype(float)
    d = (a.map(sw) - b.map(sw)).abs()
    pid = df["participant_id"]
    g1 = agree[d >= 0.25].groupby(pid[d >= 0.25]).mean()
    g2 = agree[d < 1e-9].groupby(pid[d < 1e-9]).mean()
    j = pd.concat([g1, g2], axis=1, join="inner").dropna()
    if len(j) < 3 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return 0.0
    return float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
