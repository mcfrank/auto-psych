# name: consecutive_response_perseveration
# description: P(yes | previous trial yes) minus P(yes | previous trial no) over consecutive trials (trial_index increments by 1) with a different number_set; observed > null means people perseverate on their previous answer beyond what the stimulus-driven model produces.
def test_statistic(df):
    ti = df["trial_index"].astype(int).values
    r = df["rating"].astype(float).values
    s = df["number_set"].astype(str).values
    consec = (ti[1:] == ti[:-1] + 1) & (s[1:] != s[:-1])
    prev, cur = r[:-1][consec], r[1:][consec]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
