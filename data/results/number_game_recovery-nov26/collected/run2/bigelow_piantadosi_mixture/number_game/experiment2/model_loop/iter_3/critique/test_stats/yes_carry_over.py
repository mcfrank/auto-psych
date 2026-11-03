# name: yes_carry_over
# description: Within each participant (a new one starts at trial_index 0), yes rate on trials whose previous trial (a different set) was answered yes minus yes rate after a no; observed above null means responses perseverate/carry over across trials beyond what the stimulus-driven model predicts.
def test_statistic(df):
    ti = df["trial_index"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    s = df["number_set"].astype(str).to_numpy()
    pid = np.cumsum(ti == 0)
    ok = np.zeros(len(y), bool)
    ok[1:] = (pid[1:] == pid[:-1]) & (s[1:] != s[:-1])
    prev = np.zeros(len(y))
    prev[1:] = y[:-1]
    a = ok & (prev == 1)
    b = ok & (prev == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
