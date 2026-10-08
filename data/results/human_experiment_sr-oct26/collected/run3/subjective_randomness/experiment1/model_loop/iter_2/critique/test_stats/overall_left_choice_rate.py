# name: overall_left_choice_rate
# description: Overall proportion of Left choices across all trials; observed away from null_mean means people have a side bias the model (which has no left/right term) cannot produce — above means a Left bias, below a Right bias.
def test_statistic(df):
    return float(df["chose_left"].mean())
