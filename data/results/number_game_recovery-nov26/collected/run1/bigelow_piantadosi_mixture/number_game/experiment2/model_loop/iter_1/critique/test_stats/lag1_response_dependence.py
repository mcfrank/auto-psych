# name: lag1_response_dependence
# description: P(yes | previous trial yes) minus P(yes | previous trial no) over consecutive trials of the same session (trial_index increments by 1); observed above null means responses perseverate across trials, which the trial-independent model underproduces.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].astype(float).to_numpy()
    ok = np.diff(ti) == 1
    prev = r[:-1][ok]
    cur = r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
