# name: lag1_response_repeat
# description: Proportion of consecutive trials within a participant's session that receive the same yes/no answer as the previous trial; observed above null means people perseverate on their previous answer more than the trial-independent model predicts.
def test_statistic(df):
    r = df["rating"].to_numpy().astype(int)
    ti = df["trial_index"].to_numpy()
    same_session = ti[1:] != 0
    rep = r[1:] == r[:-1]
    return float(rep[same_session].mean())
