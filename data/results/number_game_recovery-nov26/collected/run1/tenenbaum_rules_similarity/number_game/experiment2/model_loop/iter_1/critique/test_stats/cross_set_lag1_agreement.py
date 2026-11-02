# name: cross_set_lag1_agreement
# description: Proportion of consecutive trials (trial_index increases by 1, in row order) with different sets that receive the same response; observed above null means responses carry over between trials (participant bias/stickiness) that the independent-trial model lacks.
def test_statistic(df):
    r = df["rating"].values
    ti = df["trial_index"].values
    ns = df["number_set"].astype(str).values
    ok = (np.diff(ti) == 1) & (ns[1:] != ns[:-1])
    if not ok.any():
        return 0.0
    return float((r[1:] == r[:-1])[ok].mean())
