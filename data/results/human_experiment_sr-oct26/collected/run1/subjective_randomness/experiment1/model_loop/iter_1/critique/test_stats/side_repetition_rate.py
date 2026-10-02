# name: side_repetition_rate
# description: Proportion of consecutive trials (within participant, ordered by trial_index) on which the participant chose the same side as on the previous trial; observed above the null means the model misses response perseveration, below means a side-alternation habit.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    pid = d["participant_id"].to_numpy(); y = d["chose_left"].to_numpy()
    same_p = pid[1:] == pid[:-1]
    if same_p.sum() == 0:
        return 0.5
    return float((y[1:] == y[:-1])[same_p].mean())
