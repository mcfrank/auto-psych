# name: left_rate_session_drift
# description: Left-choice rate in the second half of each participant's session minus that in the first half (split at the participant's median trial_index); the model's side habit is constant over the session, so observed far from null means the Left/Right bias drifts with time (observed > null: drift toward Left late, < null: toward Right).
def test_statistic(df):
    t = df["trial_index"].to_numpy()
    med = df.groupby("participant_id")["trial_index"].transform("median").to_numpy()
    y = df["chose_left"].to_numpy()
    late = t > med
    return float(y[late].mean() - y[~late].mean())
