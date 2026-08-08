# name: generator_condition_spread
# description: Standard deviation of the mean chose_left rate across generating_model categories; probes whether the model tracks distinct generative sources (each depositing a different within-sequence structure) or stays flat because it is order-blind.
def test_statistic(df):
    means = df.groupby("generating_model")["chose_left"].mean()
    if len(means) < 2:
        return 0.0
    return float(np.nanstd(means.values))
