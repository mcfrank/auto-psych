# name: left_choice_rate
# description: Overall proportion of Left choices; t_observed > null_mean means a left-side response bias the side-symmetric model does not produce.
def test_statistic(df):
    return float(df["chose_left"].astype(float).mean())
