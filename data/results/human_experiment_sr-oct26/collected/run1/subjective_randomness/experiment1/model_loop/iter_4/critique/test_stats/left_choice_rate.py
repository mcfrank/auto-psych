# name: left_choice_rate
# description: Overall proportion of trials choosing the left sequence; observed away from the null (~0.5) means a side bias the model (which has no side term) fails to produce.
def test_statistic(df):
    return float(df["chose_left"].mean())
