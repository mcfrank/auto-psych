# name: response_side_repeat_rate
# description: Proportion of consecutive trials (within participant, ordered by trial_index) on which the participant pressed the same side as on the previous trial; observed above null_mean means response perseveration the model (trial-independent) lacks, below means more side alternation than predicted.
def test_statistic(df):
    d = df.sort_values(["participant_id", "trial_index"])
    y = d["chose_left"].to_numpy(); p = d["participant_id"].to_numpy()
    same_p = p[1:] == p[:-1]
    if same_p.sum() == 0:
        return 0.5
    return float(np.mean((y[1:] == y[:-1])[same_p]))
