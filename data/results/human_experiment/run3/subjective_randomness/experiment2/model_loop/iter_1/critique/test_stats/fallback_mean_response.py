# name: fallback_mean_response
# description: Marginal mean of the response column (the overall choice rate).
def test_statistic(df):
    return float(df["chose_left"].astype(float).mean())
