# name: lag1_response_dependence
# description: P(yes | previous trial yes) minus P(yes | previous trial no) over consecutive trials (trial_index step of 1); observed > null means people show sequential perseveration/response autocorrelation that the trial-independent model does not produce.
def test_statistic(df):
    r = df["rating"].astype(float).values
    ti = df["trial_index"].astype(int).values
    ok = np.diff(ti) == 1
    prev, cur = r[:-1][ok], r[1:][ok]
    if (prev == 1).sum() == 0 or (prev == 0).sum() == 0:
        return 0.0
    return float(cur[prev == 1].mean() - cur[prev == 0].mean())
