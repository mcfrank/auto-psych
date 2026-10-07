# name: frac_participants_prefer_fewer_switches
# description: Fraction of participants who, on pairs differing in switch count, choose the sequence with FEWER switches more than half the time; observed above null_mean means some people hold a reversed (streaky-is-random) preference that the model's shared-sign sensitivity cannot represent.
def test_statistic(df):
    def sw(s):
        return sum(x != y for x, y in zip(s, s[1:]))
    u = pd.unique(pd.concat([df["sequence_a"], df["sequence_b"]]))
    SW = {s: sw(s) for s in u}
    ka = df["sequence_a"].map(SW).values; kb = df["sequence_b"].map(SW).values
    m = ka != kb
    if m.sum() == 0:
        return 0.0
    y = df["chose_left"].values
    fewer = np.where(ka < kb, y, 1 - y).astype(float)
    s = pd.Series(fewer[m]).groupby(df["participant_id"].values[m]).mean()
    return float((s > 0.5).mean())
