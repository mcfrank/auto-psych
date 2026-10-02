# name: response_side_repetition_rate
# description: Proportion of consecutive trial pairs (within a participant, ordered by trial_index) on which the same SIDE was chosen as on the previous trial; observed above null_mean means people perseverate on a response button beyond what the trial-independent model produces (below: they over-switch sides).
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    c = d["chose_left"].to_numpy()
    p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    rep = (c[1:] == c[:-1])[same_p]
    return float(rep.mean()) if rep.size else 0.0
