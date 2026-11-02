# name: lag1_same_set_carryover
# description: Within a participant, among consecutive trials sharing the same set, P(yes | previous yes) minus P(yes | previous no); observed above null means responses perseverate (or set-level belief drifts) beyond what the stimulus-driven model produces, below null means alternation.
def test_statistic(df):
    pid = (df["trial_index"].values == 0).cumsum()
    r = df["rating"].values.astype(float)
    s = df["number_set"].values
    ok = (pid[1:] == pid[:-1]) & (s[1:] == s[:-1])
    prev = r[:-1][ok]
    cur = r[1:][ok]
    a = cur[prev == 1]
    b = cur[prev == 0]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
