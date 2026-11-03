# name: between_set_yes_rate_variance
# description: Variance across distinct number_sets of the per-set mean yes rate; observed above null means people's generalization breadth differs more sharply between sets (more all-or-none) than the model, below null means less set-specific.
def test_statistic(df):
    g = df.groupby(df["number_set"].astype(str))["rating"].mean()
    return float(np.var(g.to_numpy(dtype=float))) if len(g) > 1 else 0.0
