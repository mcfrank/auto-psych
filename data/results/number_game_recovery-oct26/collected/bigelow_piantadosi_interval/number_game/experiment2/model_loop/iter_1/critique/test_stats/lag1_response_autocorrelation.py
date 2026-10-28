# name: lag1_response_autocorrelation
# description: Pearson correlation between a response and the same participant's immediately preceding response (consecutive trial_index); observed above the null means people show sequential yes/no carryover (perseveration) the trial-independent model fails to produce, below means alternation it fails to produce.
def test_statistic(df):
    r = df["rating"].astype(float).to_numpy()
    ti = df["trial_index"].astype(int).to_numpy()
    ok = np.diff(ti) == 1
    a, b = r[:-1][ok], r[1:][ok]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
