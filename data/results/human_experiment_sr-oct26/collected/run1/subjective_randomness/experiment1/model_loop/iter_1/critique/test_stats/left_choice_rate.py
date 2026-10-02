# name: left_choice_rate
# description: Overall proportion of trials on which the left sequence was chosen; observed above the null (about 0.5 given randomised sides) means the model misses a left-side response bias.
def test_statistic(df):
    return float(df["chose_left"].astype(float).mean())
