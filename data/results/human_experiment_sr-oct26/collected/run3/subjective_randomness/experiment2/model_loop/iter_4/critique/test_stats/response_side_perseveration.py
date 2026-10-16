# name: response_side_perseveration
# description: Proportion of consecutive trials (within participant, ordered by trial_index) on which the same side (Left/Right) is chosen as on the previous trial; observed > null means the model under-produces side perseveration (sequential response dependence), < null that people alternate sides more than predicted.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].to_numpy()
    p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    return float((y[1:] == y[:-1])[same_p].mean())
