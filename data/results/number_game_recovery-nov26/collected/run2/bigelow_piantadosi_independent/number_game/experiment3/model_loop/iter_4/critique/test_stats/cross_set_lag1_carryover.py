# name: cross_set_lag1_carryover
# description: Within a participant (participants = runs starting at trial_index 0), P(yes | previous trial yes) minus P(yes | previous trial no), over consecutive trials whose number_set differs; observed above null means answers carry over between unrelated trials more than the model's independent per-trial responses (plus stable person bias) produce.
def test_statistic(df):
    r = df["rating"].to_numpy().astype(float)
    ti = df["trial_index"].to_numpy()
    s = df["number_set"].astype(str).to_numpy()
    pid = np.cumsum(ti == 0)
    ok = (pid[1:] == pid[:-1]) & (s[1:] != s[:-1])
    prev = r[:-1][ok]
    cur = r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
