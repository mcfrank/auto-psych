# name: left_choice_rate
# description: Overall proportion of trials on which the LEFT sequence was chosen; observed above the model's null_mean means people have a left-side response bias the model (which has no side term) under-produces, below means a right bias.
def test_statistic(df):
    return float(df["chose_left"].mean())
