# name: anti_alternator_share
# description: Share of participants who choose the higher-switch sequence on fewer than 40% of pairs differing in switch rate by >= 0.25; observed above null means a distinct subgroup prefers streaky sequences more than the model's population of ideal switching rates produces, below means the model over-produces such people.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    seqs = pd.unique(pd.concat([a, b]))
    sw = {s: sum(x != y for x, y in zip(s, s[1:])) / (len(s) - 1) for s in seqs}
    d = a.map(sw) - b.map(sw)
    m = d.abs() >= 0.25
    ch = pd.Series(np.where(d[m] > 0, df["chose_left"][m], 1 - df["chose_left"][m]), index=d[m].index)
    per = ch.groupby(df["participant_id"][m]).mean()
    if len(per) == 0:
        return 0.0
    return float((per < 0.4).mean())
