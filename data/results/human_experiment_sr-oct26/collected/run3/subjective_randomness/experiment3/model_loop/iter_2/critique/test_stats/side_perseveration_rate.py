# name: side_perseveration_rate
# description: Share of consecutive trials (within participant, ordered by trial_index) on which the participant clicks the same side as on the previous trial; observed above null means people perseverate on a response side more than the model (trials independent given person) produces, below means they alternate sides more than it predicts.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].values
    p = d["participant_id"].values
    same_p = p[1:] == p[:-1]
    rep = (y[1:] == y[:-1])[same_p]
    return float(rep.mean()) if rep.size else 0.5
