# name: within_set_lag1_agreement
# description: Proportion of consecutive trials (same participant, same set, adjacent trial_index) giving the same answer; observed above null means answers within a set block are more sticky/perseverative than the model's independent per-trial responses.
def test_statistic(df):
    s = df["number_set"].to_numpy()
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy()
    ok = (s[1:] == s[:-1]) & (ti[1:] == ti[:-1] + 1)
    if ok.sum() == 0:
        return 0.0
    return float(np.mean(r[1:][ok] == r[:-1][ok]))
