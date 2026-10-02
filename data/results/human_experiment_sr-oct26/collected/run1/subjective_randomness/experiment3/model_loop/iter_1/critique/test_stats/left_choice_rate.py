# name: left_choice_rate
# description: Overall proportion of trials on which the LEFT sequence was chosen; observed above null_mean means people have a left-side bias the (side-symmetric) model under-produces, below means a right-side bias.
def test_statistic(df):
    return float(df["chose_left"].mean())
