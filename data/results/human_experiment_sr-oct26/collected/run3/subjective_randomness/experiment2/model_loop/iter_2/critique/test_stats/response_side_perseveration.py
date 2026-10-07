# name: response_side_perseveration
# description: Proportion of consecutive trials (within participant, ordered by trial_index) on which the same side (Left/Right) is chosen as on the previous trial; observed > null means the model under-produces motor/side perseveration, < null that people alternate response sides more than the stimulus-driven model predicts.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].to_numpy()
    p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    rep = (y[1:] == y[:-1])[same_p]
    return float(rep.mean())
