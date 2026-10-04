# name: left_choice_rate
# description: Overall proportion of Left choices; observed away from null means a screen-side bias that the model (no side term) fails to reproduce.
def test_statistic(df):
    return float(df["chose_left"].mean())
