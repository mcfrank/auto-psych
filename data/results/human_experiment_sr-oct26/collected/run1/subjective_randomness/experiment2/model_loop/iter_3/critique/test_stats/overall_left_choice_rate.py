# name: overall_left_choice_rate
# description: Overall proportion of trials on which the left sequence was chosen; observed above the null means people have a left-side bias the model (which has no side term) under-produces, below means a right-side bias.
def test_statistic(df):
    return float(df["chose_left"].mean())
