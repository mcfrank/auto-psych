# name: lag1_response_repeat
# description: Proportion of consecutive trials within a participant (by trial_index order) on which the yes/no answer repeats the previous answer; observed > null means sequential response perseveration (answer inertia) that the trial-independent model does not produce, observed < null means alternation.
def test_statistic(df):
    pid = (df["trial_index"].to_numpy() == 0).cumsum()
    r = df["rating"].to_numpy().astype(int)
    same_p = pid[1:] == pid[:-1]
    rep = (r[1:] == r[:-1])[same_p]
    return float(rep.mean())
