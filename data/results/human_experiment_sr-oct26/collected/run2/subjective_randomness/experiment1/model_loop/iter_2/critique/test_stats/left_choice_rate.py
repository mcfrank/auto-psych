# name: left_choice_rate
# description: Overall proportion of Left choices; observed above null_mean means the model under-produces a left-position bias.
def test_statistic(df):
    return float(np.mean(df["chose_left"].to_numpy(dtype=float)))
