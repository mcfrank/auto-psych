# name: near_target_same_set_agreement
# description: Among consecutive trials of the same participant with the same example set and targets within 5 of each other, the share of pairs with identical answers; observed > null means people answer neighbouring queries more consistently (a stable sampled hypothesis or response carry-over) than the model's independent trial-by-trial Bernoulli answers produce, observed < null means less consistently.
def test_statistic(df):
    s = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(int).values
    ti = df["trial_index"].astype(int).values
    m = (s[1:] == s[:-1]) & (ti[1:] == ti[:-1] + 1) & (np.abs(t[1:] - t[:-1]) <= 5)
    if m.sum() == 0:
        return 0.5
    return float((r[1:][m] == r[:-1][m]).mean())
