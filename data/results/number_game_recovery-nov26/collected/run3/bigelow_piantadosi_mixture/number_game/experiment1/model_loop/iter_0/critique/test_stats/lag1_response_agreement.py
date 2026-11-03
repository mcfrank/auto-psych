# name: lag1_response_agreement
# description: Proportion of consecutive trials (trial_index increments by 1) whose responses are equal; observed > null means responses are more sequentially dependent (perseveration/streaks) than the model's independent per-trial responses produce.
def test_statistic(df):
    ti = df["trial_index"].to_numpy()
    r = df["rating"].to_numpy()
    if len(r) < 2:
        return 0.0
    consec = np.diff(ti) == 1
    same = r[1:] == r[:-1]
    if consec.sum() == 0:
        return 0.0
    return float(same[consec].mean())
