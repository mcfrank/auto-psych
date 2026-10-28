# name: lag1_response_dependence
# description: P(yes | previous consecutive trial yes) minus P(yes | previous trial no), using rows where trial_index increments by 1 from the previous row; observed above the replicate mean means people perseverate on their prior response (sequential dependence the independent-trial model lacks).
def test_statistic(df):
    ti = df["trial_index"].to_numpy().astype(int)
    r = df["rating"].to_numpy().astype(float)
    if len(r) < 3:
        return 0.0
    ok = ti[1:] == ti[:-1] + 1
    prev, cur = r[:-1][ok], r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
