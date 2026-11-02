# name: lag1_response_repetition
# description: Proportion of consecutive trials (adjacent rows, trial_index differing by 1, same number_set) on which the response repeats the previous response; observed above null means sequential perseveration/response inertia the stateless model does not produce, below means excess alternation.
def test_statistic(df):
    ti = df["trial_index"].astype(int).to_numpy()
    s = df["number_set"].astype(str).to_numpy()
    y = df["rating"].astype(int).to_numpy()
    m = (np.diff(ti) == 1) & (s[1:] == s[:-1])
    if not m.any():
        return 0.0
    return float(np.mean(y[1:][m] == y[:-1][m]))
