# name: same_side_as_previous
# description: Rate at which a participant picks the same side (left/right) as on their previous trial; observed above null means sequential response perseveration the model (static side bias only) under-produces, below means alternation of sides it does not capture.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    c = d["chose_left"].to_numpy()
    p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    if not same_p.any():
        return 0.5
    return float(np.mean((c[1:] == c[:-1])[same_p]))
